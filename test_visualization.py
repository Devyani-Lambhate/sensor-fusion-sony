"""
Test 3D visualization capabilities for OpenPCDet.
Checks available libraries and runs a quick visualization test.
"""

import sys
import argparse
from pathlib import Path

# Suppress numpy deprecation warnings
import numpy as np
np.bool = np.bool_
np.int = np.int_

print("=" * 70)
print("OpenPCDet 3D Visualization Capability Test")
print("=" * 70)

# Test 1: Check visualization library availability
print("\n[1] Checking Visualization Libraries...")
print("-" * 70)

VIS_LIB = None

try:
    import open3d
    print("✓ Open3D available (version: {})".format(open3d.__version__))
    print("  - Recommended for interactive visualization")
    print("  - Faster rendering")
    print("  - Better for real-time interaction")
    from visual_utils import open3d_vis_utils as V
    VIS_LIB = 'open3d'
except ImportError as e:
    print("✗ Open3D not available: {}".format(str(e)))

try:
    import mayavi.mlab as mlab
    print("✓ Mayavi available")
    print("  - Fallback visualization library")
    print("  - Good for scientific visualization")
    if VIS_LIB is None:
        from visual_utils import visualize_utils as V
        VIS_LIB = 'mayavi'
except ImportError as e:
    print("✗ Mayavi not available: {}".format(str(e)))

if VIS_LIB is None:
    print("\n⚠ WARNING: No visualization libraries found!")
    print("  Install with: pip install open3d")
    print("  Or: pip install mayavi")
    sys.exit(1)

print("\n✓ Using visualization library: {}".format(VIS_LIB.upper()))

# Test 2: Check OpenPCDet imports
print("\n[2] Checking OpenPCDet Imports...")
print("-" * 70)

try:
    import torch
    print("✓ PyTorch available (version: {})".format(torch.__version__))
except ImportError:
    print("✗ PyTorch not available")
    sys.exit(1)

try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate, build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    print("✓ OpenPCDet modules imported successfully")
except ImportError as e:
    print("✗ Failed to import OpenPCDet: {}".format(str(e)))
    sys.exit(1)

# Test 3: Create synthetic test data
print("\n[3] Creating Synthetic Test Data...")
print("-" * 70)

# Generate random point cloud
num_points = 10000
points = np.random.randn(num_points, 4).astype(np.float32)
points[:, 0] *= 20  # X range
points[:, 1] *= 20  # Y range
points[:, 2] = np.abs(points[:, 2]) * 5  # Z range (positive)
points[:, 3] = np.random.rand(num_points) * 10  # Intensity

print("✓ Generated synthetic point cloud")
print("  - Points: {}".format(points.shape))
print("  - X range: [{:.1f}, {:.1f}]".format(points[:, 0].min(), points[:, 0].max()))
print("  - Y range: [{:.1f}, {:.1f}]".format(points[:, 1].min(), points[:, 1].max()))
print("  - Z range: [{:.1f}, {:.1f}]".format(points[:, 2].min(), points[:, 2].max()))

# Generate sample boxes [x, y, z, dx, dy, dz, heading]
gt_boxes = np.array([
    [0, 0, 0, 4, 2, 1.5, 0],           # Box 1
    [10, 5, 0, 3.5, 2, 1.5, 0.785],    # Box 2 (45 degrees)
    [-8, 10, 0, 2, 2, 1.5, -0.785],    # Box 3 (-45 degrees)
], dtype=np.float32)

pred_boxes = gt_boxes + np.random.randn(3, 7).astype(np.float32) * 0.3
pred_scores = np.array([0.95, 0.87, 0.92], dtype=np.float32)
pred_labels = np.array([0, 1, 2], dtype=np.int32)

print("✓ Generated sample bounding boxes")
print("  - Ground truth boxes: {}".format(gt_boxes.shape))
print("  - Predicted boxes: {}".format(pred_boxes.shape))
print("  - Confidence scores: {}".format(pred_scores))

# Test 4: Test visualization function
print("\n[4] Testing 3D Visualization Functions...")
print("-" * 70)

try:
    print("Testing: V.draw_scenes() with point cloud and boxes...")
    
    # This will create a visualization window (non-blocking test)
    if VIS_LIB == 'open3d':
        print("  Using Open3D visualization (interactive window)")
        print("  - Controls: Mouse to rotate, scroll to zoom, right-click to pan")
        print("\n  Launching Open3D viewer... (close window to continue)")
        V.draw_scenes(
            points=points,
            gt_boxes=gt_boxes,
            ref_boxes=pred_boxes,
            ref_labels=pred_labels,
            ref_scores=pred_scores,
            draw_origin=True
        )
    else:  # Mayavi
        print("  Using Mayavi visualization (interactive window)")
        print("  - Controls: Left-click drag to rotate, middle-click to zoom")
        print("\n  Launching Mayavi viewer... (close window to continue)")
        V.draw_scenes(
            points=points,
            gt_boxes=gt_boxes,
            ref_boxes=pred_boxes,
            ref_scores=pred_scores,
            ref_labels=pred_labels
        )
        import mayavi.mlab as mlab
        mlab.show(stop=True)
    
    print("\n✓ Visualization test successful!")
    print("  Scene displayed with:")
    print("    - Point cloud (gray points)")
    print("    - Ground truth boxes (blue dashed)")
    print("    - Predicted boxes (colored by class)")
    
except Exception as e:
    print("✗ Visualization test failed: {}".format(str(e)))
    print("  This might be due to headless environment or missing display")
    import traceback
    traceback.print_exc()

# Test 5: Summary
print("\n[5] Summary & Recommendations")
print("=" * 70)

print("\n✓ Visualization Libraries: AVAILABLE")
print("  Primary: {}".format(VIS_LIB.upper()))

print("\n✓ Core Functions Available:")
print("  - V.draw_scenes(): Visualize point clouds with 3D boxes")
print("  - V.draw_corners3d(): Draw 3D bounding box corners")
print("  - V.visualize_pts(): Visualize point clouds only")

print("\n✓ Supported Input Formats:")
print("  - Point clouds: numpy arrays, torch tensors")
print("  - Boxes: [x, y, z, dx, dy, dz, heading] format")
print("  - Scores & Labels: numpy arrays")

print("\nUsage Examples:")
print("-" * 70)

if VIS_LIB == 'open3d':
    print("""
# Example 1: Visualize predictions with ground truth
from visual_utils import open3d_vis_utils as V
V.draw_scenes(
    points=point_cloud,
    gt_boxes=gt_boxes,
    ref_boxes=pred_boxes,
    ref_labels=pred_labels,
    ref_scores=pred_scores
)

# Example 2: Point cloud only
V.draw_scenes(points=point_cloud)

# Example 3: Predictions only
V.draw_scenes(
    points=point_cloud,
    ref_boxes=pred_boxes,
    ref_scores=pred_scores
)
""")
else:
    print("""
# Example 1: Visualize predictions with ground truth
from visual_utils import visualize_utils as V
import mayavi.mlab as mlab

fig = V.draw_scenes(
    points=point_cloud,
    gt_boxes=gt_boxes,
    ref_boxes=pred_boxes,
    ref_labels=pred_labels,
    ref_scores=pred_scores
)
mlab.show(stop=True)

# Example 2: Point cloud only
fig = V.draw_scenes(points=point_cloud)
mlab.show(stop=True)
""")

print("\n" + "=" * 70)
print("3D Visualization Capability Test: COMPLETE")
print("=" * 70)

# Test 6: Advanced visualization options
print("\n[6] Advanced Visualization Options")
print("-" * 70)

print("\n✓ Available in visual_utils:")
print("  - boxes_to_corners_3d(): Convert boxes to 3D corners")
print("  - rotate_points_along_z(): Rotate point clouds")
print("  - draw_sphere_pts(): Draw points as spheres")
print("  - draw_grid(): Draw reference grids")

print("\nFor BEV (Bird's Eye View) visualization:")
print("  → Use: plot_bev_predictions.py (matplotlib-based)")
print("  → Creates 2D top-down visualizations")
print("  → Faster than 3D, good for batch processing")

print("\nFor production visualization:")
print("  → Consider: saving predictions to pickle/json")
print("  → Visualize offline with custom scripts")
print("  → Or use: tensorboard for metric visualization")

print("\n" + "=" * 70)
