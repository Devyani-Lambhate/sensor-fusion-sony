"""
Enhanced 3D visualization test with real NuScenes dataset.
Loads ground truth boxes and generates predictions for visualization.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["PYVISTA_OFF_SCREEN"] = "true"   # if pyvista is used anywhere
import _init_path
import numpy as np
np.bool = np.bool_
np.int = np.int_

import argparse
import pickle
from pathlib import Path
import torch

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
#     print("✓ Using Open3D for visualization")
# except ImportError:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False
#     print("✓ Using Mayavi for visualization")

# Force Open3D
OPEN3D_FLAG = True
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    print("✅ Open3D visualization enabled")
except ImportError:
    print("❌ Open3D not found")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils


def load_nuscenes_data(data_root, split='val', num_samples=1):
    """
    Load NuScenes dataset information and point clouds.
    
    Args:
        data_root: Root directory of NuScenes dataset
        split: 'train' or 'val'
        num_samples: Number of samples to load
    
    Yields:
        (points, gt_boxes, sample_token, info_dict)
    """
    data_root = Path(data_root)
    
    # Load dataset info
    info_file = data_root / f'nuscenes_infos_10sweeps_{split}.pkl'
    if not info_file.exists():
        print(f"✗ Info file not found: {info_file}")
        return
    
    with open(info_file, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"✓ Loaded {len(infos)} samples from {split} split")
    
    # Process samples
    for idx, info in enumerate(infos[:num_samples]):
        sample_token = info['token']
        lidar_path = data_root / info['lidar_path']
        
        if not lidar_path.exists():
            print(f"⚠ LiDAR file not found: {lidar_path}")
            continue
        
        # Load point cloud
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)[:, :4]
        
        # Extract ground truth boxes
        gt_boxes = []
        gt_labels = []
        
        if 'gt_boxes' in info and 'gt_names' in info:
            gt_boxes = info['gt_boxes']  # (M, 7) [x, y, z, l, w, h, heading]
            gt_names = info['gt_names']
            
            # Convert class names to indices
            class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                          'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            gt_labels = np.array([class_names.index(name) if name in class_names else 0
                                 for name in gt_names])
        
        if len(gt_boxes) == 0:
            print(f"⚠ No ground truth boxes in sample {idx}")
            continue
        
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int32)
        
        print(f"\n✓ Sample {idx}: {sample_token}")
        print(f"  Points: {points.shape}")
        print(f"  GT Boxes: {gt_boxes.shape}")
        print(f"  GT Labels: {gt_labels}")
        
        yield points, gt_boxes, sample_token, info


def load_model(cfg_file, ckpt_path):
    """
    Load pretrained BEVFusion model.
    
    Args:
        cfg_file: Config file path
        ckpt_path: Checkpoint file path
    
    Returns:
        model: Loaded model
    """
    from pcdet.models import build_network
    from pcdet.datasets import DatasetTemplate
    
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Create a dummy dataset for model initialization
    class DummyDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names):
            super().__init__(
                dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=None, logger=None
            )
        
        def __len__(self):
            return 1
        
        def __getitem__(self, index):
            return {}
    
    dummy_dataset = DummyDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES)
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.load_params_from_file(filename=ckpt_path, logger=None, to_cpu=False)
    model.cuda()
    model.eval()
    
    print(f"✓ Loaded model from checkpoint: {ckpt_path}")
    
    return model


# def draw_scenes_with_custom_colors(points, gt_boxes=None, ref_boxes=None, ref_scores=None, 
#                                     gt_box_colors=None, ref_box_colors=None):
#     """
#     Draw 3D scenes with custom colors for boxes.
    
#     Args:
#         points: Point cloud (N, 4)
#         gt_boxes: Ground truth boxes (M, 7)
#         ref_boxes: Predicted boxes (K, 7)
#         ref_scores: Prediction scores (K,)
#         gt_box_colors: RGB colors for GT boxes (M, 3), default blue
#         ref_box_colors: RGB colors for ref boxes (K, 3), default green/red
    
#     Returns:
#         fig: Mayavi figure
#     """
#     from visual_utils.visualize_utils import visualize_pts, draw_multi_grid_range, boxes_to_corners_3d, draw_corners3d
#     import mayavi.mlab as mlab
    
#     if not isinstance(points, np.ndarray):
#         points = points.cpu().numpy()
#     if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
#         ref_boxes = ref_boxes.cpu().numpy()
#     if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
#         gt_boxes = gt_boxes.cpu().numpy()
#     if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
#         ref_scores = ref_scores.cpu().numpy()
    
#     fig = visualize_pts(points)
#     fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    
#     # Draw GT boxes with custom colors (default blue)
#     if gt_boxes is not None:
#         corners3d = boxes_to_corners_3d(gt_boxes)
#         if gt_box_colors is None:
#             gt_box_colors = np.tile([0, 0, 1], (len(gt_boxes), 1))
        
#         for i, corners in enumerate(corners3d):
#             color = tuple(gt_box_colors[i])
#             fig = draw_corners3d(np.array([corners]), fig=fig, color=color, max_num=1)
    
#     # Draw prediction boxes with custom colors
#     if ref_boxes is not None and len(ref_boxes) > 0:
#         ref_corners3d = boxes_to_corners_3d(ref_boxes)
#         if ref_box_colors is None:
#             ref_box_colors = np.tile([0, 1, 0], (len(ref_boxes), 1))
        
#         for i, corners in enumerate(ref_corners3d):
#             color = tuple(ref_box_colors[i])
#             # Pass scores as an array with single element
#             score = np.array([ref_scores[i]]) if ref_scores is not None else None
#             fig = draw_corners3d(np.array([corners]), fig=fig, color=color, cls=score, max_num=1)
    
#     mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
#     return fig


def draw_scenes_with_custom_colors(points, gt_boxes=None, ref_boxes=None, ref_scores=None, 
                                    gt_box_colors=None, ref_box_colors=None, 
                                    sample_token="sample", output_dir=None):
    """Simple but reliable 2D Bird's Eye View (BEV) + 3D projection using Matplotlib"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from pathlib import Path
    
    # Use BEV (top-down) view - most useful for LiDAR
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Plot point cloud (BEV)
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 3]/points[:, 3].max(), 
               cmap='viridis', alpha=0.6)
    
    def draw_box(ax, box, color, linewidth=2, alpha=0.8):
        """Draw 3D box as rectangle in BEV"""
        x, y, z, l, w, h, yaw = box[:7]
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        corners = corners @ rot.T
        corners += [x, y]
        
        # Close the polygon
        poly = np.vstack([corners, corners[0]])
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        
        # Center
        ax.plot(x, y, 'x', color=color, markersize=8)
    
    # Draw GT boxes - Blue
    if gt_boxes is not None:
        print('gt is not none')
        colors = gt_box_colors if gt_box_colors is not None else np.tile([0, 0, 1], (len(gt_boxes), 1))
        for i, box in enumerate(gt_boxes):
            draw_box(ax, box, color=colors[i])
    
    # Draw Predicted boxes - Green (matched) / Red (unmatched)
    if ref_boxes is not None:
        colors = ref_box_colors if ref_box_colors is not None else np.tile([0, 1, 0], (len(ref_boxes), 1))
        for i, box in enumerate(ref_boxes):
            draw_box(ax, box, color=colors[i])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'NuScenes Sample: {sample_token[:8]}\nBlue=GT | Green=Matched Pred | Red=False Pred')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Limit view to reasonable range
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    
    if output_dir:
        save_path = Path(output_dir) / f'{sample_token[:8]}_bev_nuscenes.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved BEV image: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.1):
    """
    Match predicted boxes to ground truth boxes based on IoU.
    
    Args:
        pred_boxes: Predicted boxes (N, 7) [x, y, z, dx, dy, dz, heading]
        gt_boxes: Ground truth boxes (M, 7)
        iou_threshold: IoU threshold for matching
    
    Returns:
        - matched_indices: List of matched GT indices for each prediction (-1 if no match)
        - pred_colors: RGB colors for each prediction (1,0,0)=red for missed, (0,1,0)=green for matched
        - gt_colors: RGB colors for each GT box (0,0,1)=blue for all
    """
    # Initialize outputs
    matched_indices = np.full(len(pred_boxes), -1, dtype=np.int32)
    pred_colors = np.zeros((len(pred_boxes), 3), dtype=np.float32)
    gt_colors = np.tile([0, 0, 1], (len(gt_boxes), 1)).astype(np.float32)  # Blue for all GT
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_indices, pred_colors, gt_colors
    
    # Simple distance-based matching for visualization
    # Match predictions to GT based on center distance
    from scipy.spatial.distance import cdist
    
    try:
        # Use center distance in xy plane for matching
        pred_centers = pred_boxes[:, :2]
        gt_centers = gt_boxes[:, :2]
        
        # Compute pairwise distances
        distances = cdist(pred_centers, gt_centers, metric='euclidean')
        
        # Match predictions to GT greedily based on distance
        used_gt = set()
        distance_threshold = 1.5  # meters
        
        for pred_idx in range(len(pred_boxes)):
            # Find best matching GT for this prediction
            best_dist = float('inf')
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in used_gt:
                    continue
                
                dist = distances[pred_idx, gt_idx]
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = gt_idx
            
            # If match found and distance below threshold
            if best_dist <= distance_threshold:
                matched_indices[pred_idx] = best_gt_idx
                pred_colors[pred_idx] = [0, 1, 0]  # Green for correct
                used_gt.add(best_gt_idx)
            else:
                pred_colors[pred_idx] = [1, 0, 0]  # Red for missed/incorrect
        
        print(f"    Matched predictions: {np.sum(matched_indices >= 0)}/{len(pred_boxes)}")
        
    except Exception as e:
        print(f"    ⚠ Could not compute matching: {e}")
        # If matching fails, color all predictions as red (missed)
        pred_colors[:] = [1, 0, 0]
    
    return matched_indices, pred_colors, gt_colors


def generate_predictions_from_model(model, points, info_dict, cfg):
    """
    Generate predictions using the loaded model.
    
    Args:
        model: Loaded model
        points: Point cloud (N, 4) [x, y, z, intensity]
        info_dict: Sample info dictionary
        cfg: Config object with CLASS_NAMES
    
    Returns:
        (pred_boxes, pred_scores, pred_labels)
    """
    from pcdet.models import load_data_to_gpu
    
    # Prepare input batch
    batch_dict = {
        'points': torch.from_numpy(points).float().cuda(),
        'frame_id': 0,
    }
    
    # Add dummy batch dimension and extra required fields
    batch_dict['points'] = torch.from_numpy(points).float().unsqueeze(0).cuda()
    
    try:
        # Run inference
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)
        
        if pred_dicts is None or len(pred_dicts) == 0:
            print(f"  ⚠ No predictions generated by model")
            return None, None, None
        
        # Extract predictions from first batch
        pred_dict = pred_dicts[0]
        
        pred_boxes = pred_dict.get('pred_boxes', None)
        pred_scores = pred_dict.get('pred_scores', None)
        pred_labels = pred_dict.get('pred_labels', None)
        
        if pred_boxes is None or len(pred_boxes) == 0:
            print(f"  ⚠ No predictions generated by model")
            return None, None, None
        
        # Convert to numpy
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        
        print(f"    Got {len(pred_boxes)} predictions from model")
        
        return pred_boxes, pred_scores, pred_labels
        
    except Exception as e:
        print(f"  ⚠ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_3d_visualization_nuscenes(cfg_file, data_root, ckpt_path=None, output_dir=None, num_samples=1):
    """
    Test 3D visualization with real NuScenes data and model predictions.
    
    Args:
        cfg_file: Config file path
        data_root: NuScenes data root directory
        ckpt_path: Path to model checkpoint (optional)
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    
    # Load config for class names
    cfg_from_yaml_file(cfg_file, cfg)
    class_names = cfg.CLASS_NAMES
    
    # Load model if checkpoint provided
    model = None
    if ckpt_path:
        try:
            model = load_model(cfg_file, ckpt_path)
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print(f"  Falling back to synthetic noise predictions\n")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Visualizations will be saved to: {output_dir}\n")
    else:
        print("\n✓ Visualizations will be displayed only (not saved)\n")
    
    sample_count = 0
    for points, gt_boxes, sample_token, info_dict in load_nuscenes_data(data_root, num_samples=num_samples):
        
        # Generate predictions
        if model is not None:
            print(f"  Running inference...")
            pred_boxes, pred_scores, pred_labels = generate_predictions_from_model(model, points, info_dict, cfg)
            
            if pred_boxes is None:
                print(f"  Falling back to GT-based synthetic predictions")
                # Fallback: use GT with slight noise
                pred_boxes = gt_boxes.copy()
                noise = np.random.randn(*gt_boxes.shape).astype(np.float32) * 0.05
                pred_boxes[:, :6] += noise[:, :6] * 0.3
                pred_scores = np.full(len(gt_boxes), 0.95, dtype=np.float32)
                pred_labels = np.arange(len(gt_boxes), dtype=np.int32)
        else:
            print(f"  ⚠ No model checkpoint provided, using synthetic predictions for testing")
            # For testing without model: create synthetic predictions from GT with small noise
            pred_boxes = gt_boxes.copy()
            noise = np.random.randn(*gt_boxes.shape).astype(np.float32) * 0.05
            pred_boxes[:, :6] += noise[:, :6] * 0.2
            pred_scores = np.full(len(gt_boxes), 0.90, dtype=np.float32)
            pred_labels = np.arange(len(gt_boxes), dtype=np.int32)
        
        print(f"\n  Predictions:")
        print(f"    Pred Boxes: {pred_boxes.shape}")
        print(f"    Scores: {pred_scores}")
        print(f"    Labels: {pred_labels}")
        
        # Match predictions to GT and get colors
        print(f"  Matching predictions to ground truth...")
        matched_indices, pred_colors, gt_colors = match_predictions_to_gt(pred_boxes, gt_boxes)
        
        # Create visualization with custom colors
        print(f"  Creating 3D visualization...")
        # fig = draw_scenes_with_custom_colors(
        #     points=points,
        #     gt_boxes=gt_boxes,
        #     ref_boxes=pred_boxes,
        #     ref_scores=pred_scores,
        #     gt_box_colors=gt_colors,
        #     ref_box_colors=pred_colors
        # )
        
        fig = draw_scenes_with_custom_colors(
            points=points,
            gt_boxes=gt_boxes,
            ref_boxes=pred_boxes,
            ref_scores=pred_scores,
            gt_box_colors=gt_colors,
            ref_box_colors=pred_colors,
            sample_token=sample_token,
            output_dir=output_dir
        )

        sample_count += 1
        
        # Save or display
        # if output_dir and not OPEN3D_FLAG:
        #     save_path = output_dir / f'{sample_token[:8]}_nuscenes.png'
        #     mlab.savefig(str(save_path))
        #     print(f"  ✓ Saved to: {save_path}")
        #     mlab.close(fig)
        # elif OPEN3D_FLAG:
        #     print(f"  ✓ Open3D visualization (interactive)")
        #     break  # Open3D blocks, so only do one
        # else:
        #     print(f"  ✓ Displaying Mayavi visualization")
        #     mlab.show(stop=True)

        if output_dir:
            saved_files = list(output_dir.glob('*_bev_nuscenes.png'))
            print(f"\n✓ Total visualizations saved: {len(saved_files)}")
            for f in sorted(saved_files)[:10]:
                print(f"  - {f.name}")
        
        # sample_count += 1
    
    print("\n" + "=" * 70)
    print("3D Visualization Test Complete!")
    print("=" * 70)
    
    if output_dir:
        saved_files = list(output_dir.glob('*_nuscenes.png'))
        print(f"\n✓ Total visualizations saved: {len(saved_files)}")
        for f in sorted(saved_files)[:10]:
            print(f"  - {f.name}")
        if len(saved_files) > 10:
            print(f"  ... and {len(saved_files) - 10} more")
    
    return sample_count


def test_synthetic_data(output_dir=None, num_samples=1):
    """
    Fallback test with synthetic data if NuScenes data not available.
    
    Args:
        output_dir: Directory to save visualizations
        num_samples: Number of test samples to generate
    """
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Visualizations will be saved to: {output_dir}\n")
    else:
        print("\n✓ Visualizations will be displayed only (not saved)\n")
    
    for sample_idx in range(num_samples):
        print(f"Generating test sample {sample_idx + 1}/{num_samples}...")
        
        # Generate synthetic point cloud
        np.random.seed(42 + sample_idx)
        num_points = 5000
        points = np.random.randn(num_points, 4).astype(np.float32)
        points[:, 0] *= 30  # X range
        points[:, 1] *= 30  # Y range
        points[:, 2] = np.abs(points[:, 2]) * 3  # Z range (positive)
        points[:, 3] = np.random.rand(num_points) * 10  # Intensity
        
        # Generate ground truth boxes [x, y, z, dx, dy, dz, heading]
        gt_boxes = np.array([
            [0, 0, 0, 4, 2, 1.5, 0],
            [15, 10, 0, 3.5, 2, 1.5, 0.785],
            [-15, 15, 0, 2, 2, 1.5, -0.785],
        ], dtype=np.float32)
        
        # Generate predicted boxes (slightly noisy)
        pred_boxes = gt_boxes + np.random.randn(3, 7).astype(np.float32) * 0.5
        pred_scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)
        pred_labels = np.array([0, 1, 2], dtype=np.int32)
        
        print(f"  Points: {points.shape}")
        print(f"  GT Boxes: {gt_boxes.shape}")
        print(f"  Pred Boxes: {pred_boxes.shape}")
        print(f"  Scores: {pred_scores}")
        
        # Create visualization
        print(f"  Creating 3D visualization...")
        fig = V.draw_scenes(
            points=points,
            gt_boxes=gt_boxes,
            ref_boxes=pred_boxes,
            ref_labels=pred_labels,
            ref_scores=pred_scores
        )
        
        # Save or display
        if output_dir and not OPEN3D_FLAG:
            save_path = output_dir / f'sample_{sample_idx:04d}.png'
            mlab.savefig(str(save_path))
            print(f"  ✓ Saved to: {save_path}")
            mlab.close(fig)
        elif OPEN3D_FLAG:
            print(f"  ✓ Open3D visualization (interactive)")
            break  # Open3D blocks, so only do one
        else:
            print(f"  ✓ Displaying Mayavi visualization")
            mlab.show(stop=True)
    
    print("\n" + "=" * 70)
    print("3D Visualization Test Complete!")
    print("=" * 70)
    
    if output_dir:
        saved_files = list(output_dir.glob('*.png'))
        print(f"\n✓ Total visualizations saved: {len(saved_files)}")
        for f in saved_files[:5]:
            print(f"  - {f.name}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description='Test 3D visualization with NuScenes data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all NuScenes samples with real model predictions
  python test_3d_visualization.py --cfg_file cfgs/nuscenes_models/bevfusion.yaml \\
      --data_root ../data/nuscenes/v1.0-mini \\
      --ckpt ../output/nuscenes_models/bevfusion/test_run_lr_weights/ckpt/checkpoint_epoch_2.pth \\
      --output_dir ./viz_output
  
  # Visualize first 10 samples
  python test_3d_visualization.py --cfg_file cfgs/nuscenes_models/bevfusion.yaml \\
      --data_root ../data/nuscenes/v1.0-mini --output_dir ./viz_output --num_samples 10
  
  # Display visualization only (interactive)
  python test_3d_visualization.py --cfg_file cfgs/nuscenes_models/bevfusion.yaml \\
      --data_root ../data/nuscenes/v1.0-mini --num_samples 1
        """
    )
    
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/bevfusion.yaml',
                       help='Config file for NuScenes model')
    parser.add_argument('--data_root', type=str, default='../data/nuscenes/v1.0-mini',
                       help='Root directory of NuScenes dataset')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Path to model checkpoint for predictions')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization screenshots')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of test samples to generate (-1 for all samples)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("OpenPCDet 3D Visualization Test")
    print("=" * 70)
    print(f"Using library: {'Open3D' if OPEN3D_FLAG else 'Mayavi'}")
    if args.num_samples == -1:
        print(f"Num samples: ALL")
    else:
        print(f"Num samples: {args.num_samples}")
    if args.ckpt:
        print(f"Checkpoint: {args.ckpt}")
    
    print(f"Mode: NuScenes dataset")
    print(f"Data root: {args.data_root}")
    
    # Check if NuScenes data exists
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"✗ Data root not found: {data_root}")
        return
    
    # If num_samples is -1, use a very large number to load all samples
    num_to_load = args.num_samples if args.num_samples > 0 else 999999
    
    samples_loaded = test_3d_visualization_nuscenes(
        cfg_file=args.cfg_file,
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        num_samples=num_to_load
    )


if __name__ == '__main__':
    main()
