import torch
import numpy as np
import argparse
from multiprocessing import Pool
from terminaltables import AsciiTable
from mmcv.utils import print_log

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from shapely.geometry import Polygon
from postprocess import *


def compute_reg_nll(dets, gt, covar_e=None, covar_a=None, w=0.0):
    assert len(dets) == len(gt)
    dets_loc = torch.from_numpy(dets[:, :8])
    dets_loc = torch.reshape(dets_loc, (-1, 2))
    gt_loc = torch.from_numpy(gt)
    gt_loc = torch.reshape(gt_loc, (-1, 2))
    if len(dets) == 0:
        return None
    if len(dets[0]) <= 9:
        covar_matrix = covar_e
    else:
        dets_covar = torch.from_numpy(dets[:, 9:])
        dets_covar = torch.reshape(dets_covar, (-1, 3))
        u_matrix = torch.zeros((dets_covar.shape[0], 2, 2))
        u_matrix[:, 0, 0] = torch.exp(dets_covar[:, 0])
        u_matrix[:, 0, 1] = dets_covar[:, 1]
        u_matrix[:, 1, 1] = torch.exp(dets_covar[:, 2])
        sigma_inverse = torch.matmul(torch.transpose(u_matrix, 1, 2), u_matrix)
        covar_matrix = torch.linalg.inv(sigma_inverse)
        if covar_e is not None and covar_a is not None:
            covar_matrix = covar_e + (0.5 * covar_a + 0.5 * covar_matrix) * 100.0
        else:
            covar_matrix = covar_matrix * 100.0
    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        dets_loc, covariance_matrix=covar_matrix
    )
    negative_log_prob = -predicted_multivariate_normal_dists.log_prob(gt_loc)
    return negative_log_prob.tolist()


def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(
    det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None
):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (
            np.zeros(gt_bboxes.shape[0], dtype=np.bool),
            np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool),
        )
    )
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1]
            )
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    gt_corners = np.zeros((gt_bboxes.shape[0], 4, 2), dtype=np.float32)
    pred_corners = np.zeros((det_bboxes.shape[0], 4, 2), dtype=np.float32)

    for k in range(gt_bboxes.shape[0]):
        gt_corners[k, 0, 0] = gt_bboxes[k][0]
        gt_corners[k, 0, 1] = gt_bboxes[k][1]
        gt_corners[k, 1, 0] = gt_bboxes[k][2]
        gt_corners[k, 1, 1] = gt_bboxes[k][3]
        gt_corners[k, 2, 0] = gt_bboxes[k][4]
        gt_corners[k, 2, 1] = gt_bboxes[k][5]
        gt_corners[k, 3, 0] = gt_bboxes[k][6]
        gt_corners[k, 3, 1] = gt_bboxes[k][7]

    if det_bboxes.ndim == 1:
        det_bboxes = np.array([det_bboxes])

    for k in range(det_bboxes.shape[0]):
        pred_corners[k, 0, 0] = det_bboxes[k][0]
        pred_corners[k, 0, 1] = det_bboxes[k][1]
        pred_corners[k, 1, 0] = det_bboxes[k][2]
        pred_corners[k, 1, 1] = det_bboxes[k][3]
        pred_corners[k, 2, 0] = det_bboxes[k][4]
        pred_corners[k, 2, 1] = det_bboxes[k][5]
        pred_corners[k, 3, 0] = det_bboxes[k][6]
        pred_corners[k, 3, 1] = det_bboxes[k][7]

    gt_box = convert_format(gt_corners)
    pred_box = convert_format(pred_corners)
    save_flag = False
    for gt in gt_box:
        iou = np.array(compute_iou(gt, pred_box))
        if not save_flag:
            box_iou = iou
            save_flag = True
        else:
            box_iou = np.vstack((box_iou, iou))

    # make dimension the same
    if len(gt_box) == 1:
        box_iou = np.array([box_iou])

    ious = box_iou.T
    #    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)

    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, 8])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1]
            )
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def match_tp_fp(
    det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None
):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (
            np.zeros(gt_bboxes.shape[0], dtype=np.bool),
            np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool),
        )
    )
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
        
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.int)
    fp = np.zeros((num_scales, num_dets), dtype=np.int)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1]
            )
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    gt_corners = np.zeros((gt_bboxes.shape[0], 4, 2), dtype=np.float32)
    pred_corners = np.zeros((det_bboxes.shape[0], 4, 2), dtype=np.float32)

    for k in range(gt_bboxes.shape[0]):
        gt_corners[k, 0, 0] = gt_bboxes[k][0]
        gt_corners[k, 0, 1] = gt_bboxes[k][1]
        gt_corners[k, 1, 0] = gt_bboxes[k][2]
        gt_corners[k, 1, 1] = gt_bboxes[k][3]
        gt_corners[k, 2, 0] = gt_bboxes[k][4]
        gt_corners[k, 2, 1] = gt_bboxes[k][5]
        gt_corners[k, 3, 0] = gt_bboxes[k][6]
        gt_corners[k, 3, 1] = gt_bboxes[k][7]

    if det_bboxes.ndim == 1:
        det_bboxes = np.array([det_bboxes])

    for k in range(det_bboxes.shape[0]):
        pred_corners[k, 0, 0] = det_bboxes[k][0]
        pred_corners[k, 0, 1] = det_bboxes[k][1]
        pred_corners[k, 1, 0] = det_bboxes[k][2]
        pred_corners[k, 1, 1] = det_bboxes[k][3]
        pred_corners[k, 2, 0] = det_bboxes[k][4]
        pred_corners[k, 2, 1] = det_bboxes[k][5]
        pred_corners[k, 3, 0] = det_bboxes[k][6]
        pred_corners[k, 3, 1] = det_bboxes[k][7]

    gt_box = convert_format(gt_corners)
    pred_box = convert_format(pred_corners)
    save_flag = False
    for gt in gt_box:
        iou = np.array(compute_iou(gt, pred_box))
        if not save_flag:
            box_iou = iou
            save_flag = True
        else:
            box_iou = np.vstack((box_iou, iou))

    # make dimension the same
    if len(gt_box) == 1:
        box_iou = np.array([box_iou])

    ious = box_iou.T
    #    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)

    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, 8])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1]
            )
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = matched_gt
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def eval_nll(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5,
    logger=None,
    nproc=4,
    covar_e=None,
    covar_a=None,
    w=0.0
):
    """Your original eval_nll function"""
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = (
        [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges] if scale_ranges is not None else None
    )

    pool = Pool(nproc)
    eval_results = []

    for i in range(num_classes):
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        tp_nll = []
        tpfp_func = match_tp_fp

        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_gts,
                cls_gts_ignore,
                [iou_thr] * num_imgs,
                [area_ranges] * num_imgs,
            ),
        )
        tp_all, fp_all = list(zip(*tpfp))

        for dets, gt, match, fp in zip(cls_dets, cls_gts, tp_all, fp_all):
            tp = np.squeeze((1 - fp).astype(bool))
            fp = np.squeeze(fp.astype(bool))
            match = np.squeeze(match)

            if tp.shape == ():
                tp = np.array([tp])
                match = np.array([match])

            if len(tp) != 0:
                tp_dets = dets[tp]
                tp_match = match[tp]
                tp_gt = gt[tp_match]
                nll = compute_reg_nll(tp_dets, tp_gt, covar_e, covar_a, w)
                if nll is not None:
                    tp_nll.extend(nll)

        eval_results.append({
            "num_gts": len(cls_gts[0]) if len(cls_gts) > 0 else 0,
            "NLL": np.mean(tp_nll) if tp_nll else float('nan'),
        })

    pool.close()
    return eval_results


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.
    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann["labels"] == class_id

        if ann["bboxes"].ndim == 1:
            cls_gts.append(ann["bboxes"][0])
        else:
            cls_gts.append(ann["bboxes"][gt_inds, :])

        if ann.get("labels_ignore", None) is not None:
            ignore_inds = ann["labels_ignore"] == class_id

            if ann["bboxes_ignore"].ndim == 1:
                cls_gts_ignore.append(ann["bboxes_ignore"][0])
            else:
                cls_gts_ignore.append(ann["bboxes_ignore"][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 8), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


# ========================== NEW HELPER FUNCTIONS ==========================

def convert_3d_boxes_to_corners(boxes):
    """Convert 3D boxes to 8 corner coordinates (x1y1x2y2x3y3x4y4)"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 3]
    l = boxes[:, 4]
    yaw = np.arctan2(boxes[:, 6], boxes[:, 7])

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.stack([[c, -s], [s, c]], axis=1)  # (N, 2, 2)

    local = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    local = local * np.stack([w, l], axis=1)[:, None, :]

    corners = np.einsum('bij,bjk->bik', rot, local)
    corners += np.stack([x, y], axis=1)[:, None, :]

    return corners.reshape(-1, 8)


def prepare_for_nll(final_box_dicts, use_doublem=True):
    det_results = []
    for box_dict in final_box_dicts:
        boxes_3d = box_dict['pred_boxes'].cpu().numpy()
        scores = box_dict['pred_scores'].cpu().numpy()
        
        corners = boxes3d_to_corners_2d(boxes_3d)           # ← Use this
        det = np.concatenate([corners.reshape(-1, 8), scores[:, None]], axis=1)
        
        # Add uncertainty (same as before)
        if use_doublem and 'pred_uncertainty' in box_dict:
            unc = box_dict['pred_uncertainty'].cpu().numpy()
            covar = np.stack([np.log(unc**2 + 1e-6), np.zeros_like(unc), np.log(unc**2 + 1e-6)], axis=1)
            det = np.concatenate([det, covar], axis=1)
        
        det_results.append(det)
    
    return [det_results]   # structure expected by eval_nll


# ========================== MAIN ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--doublem_ckpt', type=str, default=None)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()

    if args.doublem_ckpt:
        model.model_cfg.USE_DOUBLE_M = True
        model.model_cfg.DOUBLE_M_CKPT = args.doublem_ckpt
        print(f"✅ Loaded Double-M checkpoint: {args.doublem_ckpt}")

    test_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=4,
        logger=logger,
        training=False
    )

    det_results = []
    annotations = []

    with torch.no_grad():
        for batch_dict in common_utils.progress_bar(test_loader):
            common_utils.load_data_to_gpu(batch_dict)
            model(batch_dict)

            final_box_dicts = batch_dict['final_box_dicts']
            det_results.extend(prepare_for_nll(final_box_dicts, use_doublem=bool(args.doublem_ckpt)))

            # GT
            gt_boxes = batch_dict['gt_boxes'].cpu().numpy()
            gt_corners = convert_3d_boxes_to_corners(gt_boxes[:, :7])
            annotations.append({
                "bboxes": gt_corners,
                "labels": (gt_boxes[:, -1] - 1).astype(np.int32),
            })

    # Evaluate NLL
    eval_results = eval_nll(
        det_results=det_results,
        annotations=annotations,
        iou_thr=args.iou_thr,
        logger=logger,
        covar_e=None,
        covar_a=None
    )

    # Summary
    nll_scores = [r["NLL"] for r in eval_results if not np.isnan(r.get("NLL", np.nan))]
    mean_nll = np.mean(nll_scores) if nll_scores else np.nan

    print_log("\n" + "="*70, logger=logger)
    print_log(f"FINAL NLL EVALUATION (IoU={args.iou_thr})", logger=logger)
    print_log(f"Mean NLL: {mean_nll:.4f}   (Lower is better)", logger=logger)
    print_log("="*70, logger=logger)

    table = AsciiTable([["Class", "NLL"]] + [[i, f"{r['NLL']:.4f}"] for i, r in enumerate(eval_results)])
    print_log(table.table, logger=logger)


if __name__ == "__main__":
    main()