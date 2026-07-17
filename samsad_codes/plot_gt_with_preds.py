import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from tqdm import tqdm


INFO_PKL = "data/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl"
RESULTS_PKL = "output/nuscenes_models/bevfusion/default/eval/epoch_no_number/val/original_dataset/result.pkl"
LIDAR_DIR = "data/v1.0-trainval"
SAVE_DIR = "output/nuscenes_models/bevfusion/default/eval/epoch_no_number/val/original_dataset/preds"

os.makedirs(SAVE_DIR, exist_ok=True)

POINT_LIMIT = 50000
SCORE_THRESHOLD = 0.2
TARGET_CLASS = "car"


def load_data():
    with open(INFO_PKL, "rb") as f:
        infos = pickle.load(f)

    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)

    return infos, results


def load_points(info):
    pc_path = os.path.join(LIDAR_DIR, info["lidar_path"])
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
    return points


def get_pred_boxes(sample):
    boxes = sample["boxes_lidar"]
    scores = sample["score"]
    labels = sample["name"]
    return boxes, scores, labels


def box_to_bev_corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]

    c = np.cos(yaw)
    s = np.sin(yaw)

    corners = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
    ])

    rot = np.array([
        [c, -s],
        [s,  c]
    ])

    corners = corners @ rot.T
    corners += np.array([x, y])

    return corners


def draw_boxes(ax, boxes, color, label=None, linewidth=2):
    for i, box in enumerate(boxes):
        corners = box_to_bev_corners(box)

        poly = Polygon(
            corners,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=label if i == 0 else None
        )
        ax.add_patch(poly)


def plot_bev(points, gt_boxes, pred_boxes, sample_idx, save_dir=None, sample_token=None):
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=2.0,
        c="gray",
        alpha=0.4
    )

    draw_boxes(ax, gt_boxes, color="green", label="GT")
    draw_boxes(ax, pred_boxes, color="red", label="Pred")

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal")

    ax.set_xlabel("X (meters)", fontsize=24)
    ax.set_ylabel("Y (meters)", fontsize=24)
    ax.tick_params(labelsize=20)

    ax.legend(fontsize=20)
    ax.grid(True)

    if save_dir:
        if sample_token is not None:
            plt.savefig(f"{save_dir}/gt_preds_sample_{sample_idx}_{sample_token}.png", dpi = 300)
        else:
            plt.savefig(f"{save_dir}/gt_preds_sample_{sample_idx}.png", dpi = 300)
    else:
        plt.show()

    plt.close()


def plot_single(infos, results, sample_idx, save_dir=None):
    info = infos[sample_idx]
    pred_sample = results[sample_idx]

    assert info["token"] == pred_sample["metadata"]["token"], \
        f"Token mismatch at index {sample_idx}"
    
    sample_token = info["token"]

    points = load_points(info)

    if points.shape[0] > POINT_LIMIT:
        idx = np.random.choice(
            points.shape[0],
            POINT_LIMIT,
            replace=False
        )
        points = points[idx]

    gt_boxes = info["gt_boxes"]
    gt_names = info["gt_names"]

    gt_mask = gt_names == TARGET_CLASS
    gt_boxes = gt_boxes[gt_mask]

    pred_boxes, pred_scores, pred_names = get_pred_boxes(pred_sample)

    pred_mask = (
        (pred_scores > SCORE_THRESHOLD) &
        (pred_names == TARGET_CLASS)
    )

    pred_boxes = pred_boxes[pred_mask]

    plot_bev(
        points,
        gt_boxes,
        pred_boxes,
        sample_idx,
        save_dir=save_dir,
        sample_token = sample_token
    )


def plot_all(infos, results, save_dir=None):
    for i in tqdm(range(len(infos)), desc="Plotting samples"):
        plot_single(infos, results, i, save_dir=save_dir)


def main():
    infos, results = load_data()

    # infos = infos[:100]
    # results = results[:100]

    plot_all(infos, results, save_dir=SAVE_DIR)


if __name__ == "__main__":
    main()