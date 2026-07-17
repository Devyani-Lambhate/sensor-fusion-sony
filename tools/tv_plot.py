import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import argparse
from pathlib import Path

"""
Script to generate uncertainty correlation plots (like Fig. 3.5 in the paper)
for your BEVFusion / TransFusion single-stage detector.
https://www.uni-hildesheim.de/gitlab/amirim/thesis/-/raw/main/papers/Dissertation_Di_Feng.pdf

It loads statistics_epoch_*.pth files and creates scatter plots of pairwise
uncertainties (center, height, dim, rot, vel) with PCC values.

Usage:
    python plot_uncertainty_correlations.py --stats_path ./output/cs/statistics_epoch_007.pth
    # or to plot the latest:
    python plot_uncertainty_correlations.py --stats_path ./output/cs/ --latest
"""


def load_full_uncertainties(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*_full.pth"))
        stats_file = files[-1]
        print(f"✅ Loading latest: {stats_file.name}")
    else:
        stats_file = path
        print(f"✅ Loading: {stats_file.name}")

    data = torch.load(stats_file, map_location='cpu', weights_only=True)
    unc = data.get('full_uncertainties', {})
    print(f"Number of predictions: {data.get('num_predictions', 0):,}")
    return unc


def compute_tv(sigma_tensor, channels):
    """Compute Total Variation per object (sum of variances)."""
    if len(sigma_tensor) == 0:
        return np.array([])
    num_objects = len(sigma_tensor) // channels
    sigma_reshaped = sigma_tensor.view(num_objects, channels)
    return (sigma_reshaped ** 2).sum(dim=1).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_path', type=str, default='/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet_vishal/output/nuscenes_models/bevfusion/fog_finetuned_mixed/eval/eval_with_train/epoch_15/val',
                        help='Path to statistics_epoch_XXX_full.pth or directory')
    parser.add_argument('--save_dir', type=str, default='/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet_vishal/output/nuscenes_models/bevfusion/fog_finetuned_mixed/statistics/uncertainty_plots_snowy')
    args = parser.parse_args()

    unc = load_full_uncertainties(args.stats_path)

    channels = {'center': 2, 'height': 1, 'dim': 3, 'rot': 2, 'vel': 2}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ====================== 1. Pairwise Scatter Plots + Combined Grid ======================
    pairs = [
        ('center', 'dim'), ('center', 'rot'), ('dim', 'rot'),
        ('center', 'height'), ('rot', 'vel'), ('dim', 'vel')
    ]

    pcc_table = {}
    print("\nPCC TABLE (Total Variation per object)")
    print("="*60)

    for name1, name2 in pairs:
        key1 = f"{name1}_scale"
        key2 = f"{name2}_scale"
        if key1 in unc and key2 in unc:
            tv1 = compute_tv(unc[key1], channels[name1])
            tv2 = compute_tv(unc[key2], channels[name2])
            pcc, _ = pearsonr(tv1, tv2)
            pcc = round(pcc, 3)
            pcc_table[f"{name1} vs {name2}"] = pcc
            print(f"{name1:6s} vs {name2:6s} : PCC = {pcc}")

    # Combined 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for idx, (name1, name2) in enumerate(pairs):
        key1 = f"{name1}_scale"
        key2 = f"{name2}_scale"
        if key1 in unc and key2 in unc:
            tv1 = compute_tv(unc[key1], channels[name1])
            tv2 = compute_tv(unc[key2], channels[name2])
            pcc = pcc_table[f"{name1} vs {name2}"]
            axes[idx].scatter(tv1, tv2, s=6, alpha=0.6, c='blue', edgecolors='none')
            axes[idx].set_xlabel(f'TV {name1.capitalize()}')
            axes[idx].set_ylabel(f'TV {name2.capitalize()}')
            axes[idx].set_title(f'{name1.capitalize()} vs {name2.capitalize()}\nPCC = {pcc}')
            axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "all_pairs_grid.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ====================== 2. TV Histograms ======================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for idx, name in enumerate(['center', 'height', 'dim', 'rot', 'vel']):
        key = f"{name}_scale"
        if key in unc and len(unc[key]) > 0:
            tv = compute_tv(unc[key], channels[name])
            axes[idx].hist(tv, bins=100, alpha=0.8, color='blue', edgecolor='black')
            axes[idx].set_xlabel(f'TV {name.capitalize()}')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'TV Distribution - {name.capitalize()}')
            axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "tv_histograms.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ====================== 3. Orientation Uncertainty (Fig. 3.12 style) ======================
    if 'rot_scale' in unc and 'rot_angle_deg_scale' in unc:
        print("\nGenerating orientation uncertainty plots...")
        sigma_rot = unc['rot_scale']
        theta_deg = unc['rot_angle_deg_scale'].numpy()

        # FIXED: Use compute_tv for per-object TV
        tv_rot = compute_tv(unc['rot_scale'], channels['rot'])

        # Minimum angular difference to cardinal directions (0°, 90°, 180°, 270°)
        # This is exactly how the paper computes "orientation difference"
        base_angles = np.array([0, 90, 180, 270])
        diff_to_base = np.minimum.reduce([np.abs(theta_deg - b) for b in base_angles])
        diff_to_base = np.minimum(diff_to_base, 360 - diff_to_base)

        # === THIS IS WHAT THEY CALCULATE IN THE PAPER ===
        # Overall Pearson correlation coefficient (PCC) between orientation difference and TV
        # (this gives the 0.99 you see in the image)
        pcc, _ = pearsonr(diff_to_base, tv_rot)
        pcc = round(pcc, 2)
        print(f"    → PCC (orientation difference vs TV) = {pcc:.2f}")

        # (a) Polar Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        theta_rad = np.deg2rad(theta_deg)
        ax.scatter(theta_rad, tv_rot, s=8, alpha=0.6, c='blue', edgecolors='none')
        ax.set_title("Aleatoric Uncertainty in Object Orientation\n(Radial = TV, Angular = Predicted θ)", fontsize=14)
        ax.set_rlabel_position(45)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        for angle in [0, 90, 180, 270]:
            ax.plot([np.deg2rad(angle)]*2, [0, tv_rot.max()*1.1], color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.tight_layout()
        plt.savefig(save_dir / "orientation_polar.png", dpi=400, bbox_inches='tight')
        plt.show()

        # (b) Average Uncertainty vs Orientation Difference
        base_angles = np.array([0, 90, 180, 270])
        diff_to_base = np.minimum.reduce([np.abs(theta_deg - b) for b in base_angles])
        diff_to_base = np.minimum(diff_to_base, 360 - diff_to_base)

        bins = np.arange(0, 91, 5)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        avg_tv = []
        for i in range(len(bins)-1):
            mask = (diff_to_base >= bins[i]) & (diff_to_base < bins[i+1])
            avg_tv.append(tv_rot[mask].mean() if np.any(mask) else np.nan)
        avg_tv = np.array(avg_tv)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(bin_centers, avg_tv, marker='o', linestyle='-', color='blue', linewidth=2.5)
        ax.set_xlabel('Orientation difference (degree)')
        ax.set_ylabel('Average TV (uncertainty)')
        ax.set_title(f'PCC = {pcc}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "orientation_vs_diff.png", dpi=400, bbox_inches='tight')
        plt.show()

    # ====================== Extract TV for x, y, z ======================
    # Center has 2 channels: x and y
    if 'center_scale' in unc:
        center = unc['center_scale']
        num_objects = len(center) // 2
        center_reshaped = center.view(num_objects, 2)
        tv_x = (center_reshaped[:, 0] ** 2).numpy()      # x-axis (forward)
        tv_y = (center_reshaped[:, 1] ** 2).numpy()      # y-axis (lateral)

    # Height is used as z-axis proxy
    if 'height_scale' in unc:
        tv_z = (unc['height_scale'] ** 2).numpy()

    # Use x-coordinate as distance proxy (forward direction in nuScenes)
    distance = center_reshaped[:, 0].numpy() if 'center_scale' in unc else None

    if distance is None:
        print("No center information found!")
        return

    # ====================== Plot: Aleatoric Uncertainty vs Distance ======================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # X-axis
    pcc_x, _ = pearsonr(distance, tv_x)
    axes[0].scatter(
        distance,
        tv_x,
        color='tab:blue',
        s=20,        # marker size
        alpha=0.6
    )
    axes[0].set_ylabel('TV (x-axis)')
    axes[0].set_title(f'Aleatoric uncertainty x-axis, PCC={pcc_x:.2f}')
    axes[0].grid(True, alpha=0.3)

    # Y-axis
    pcc_y, _ = pearsonr(distance, tv_y)
    axes[1].scatter(
        distance,
        tv_y,
        color='tab:blue',
        s=20,        # marker size
        alpha=0.6
    )
    axes[1].set_ylabel('TV (y-axis)')
    axes[1].set_title(f'Aleatoric uncertainty y-axis, PCC={pcc_y:.2f}')
    axes[1].grid(True, alpha=0.3)

    # Z-axis (height)
    pcc_z, _ = pearsonr(distance, tv_z)
    axes[2].scatter(
        distance,
        tv_z,
        color='tab:blue',
        s=20,        # marker size
        alpha=0.6
    )
    axes[2].set_xlabel('Distance (center x-coordinate from ego)')
    axes[2].set_ylabel('TV (z-axis / height)')
    axes[2].set_title(f'Aleatoric uncertainty z-axis, PCC={pcc_z:.2f}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "aleatoric_uncertainty_vs_distance.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ====================== 4. Error vs Uncertainty for ALL Heads ======================
    print("\nGenerating Combined Grid: Error vs Uncertainty for all heads...")

    head_info = {
        'center': {'gt_key': 'center_gt_scale', 'channels': 2, 'name': 'Center'},
        'height': {'gt_key': 'height_gt_scale', 'channels': 1, 'name': 'Height'},
        'dim':    {'gt_key': 'dim_gt_scale',    'channels': 3, 'name': 'Dimension'},
        'rot':    {'gt_key': 'rot_gt_deg_scale', 'channels': 2, 'name': 'Rotation'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (head, info) in enumerate(head_info.items()):
        scale_key = f"{head}_scale"
        gt_key    = info['gt_key']

        if scale_key not in unc or gt_key not in unc:
            axes[idx].text(0.5, 0.5, f'Missing data for {head}', ha='center', va='center', fontsize=12)
            continue

        # Predicted values
        pred_values = unc[scale_key].numpy().reshape(-1, info['channels'])

        # GT values - slice correctly to match prediction channels
        gt_values = unc[gt_key].numpy()
        if len(gt_values.shape) == 1:
            gt_values = gt_values.reshape(-1, 1)

        if head == 'center':
            gt_values = gt_values[:, 0:2]          # x, y only
        elif head == 'height':
            gt_values = gt_values[:, 2:3] if gt_values.shape[1] >= 3 else gt_values  # z only

        # Align lengths
        min_len = min(len(pred_values), len(gt_values))
        pred_values = pred_values[:min_len]
        gt_values   = gt_values[:min_len]

        # Compute error
        if head == 'rot':
            theta_pred = unc['rot_angle_deg_scale'].numpy()[:min_len].reshape(-1, 1)
            error = np.abs(theta_pred.flatten() - gt_values.flatten())
        else:
            error = np.linalg.norm(pred_values - gt_values, axis=1)

        # Uncertainty (TV)
        tv = compute_tv(unc[scale_key], channels=info['channels'])[:min_len]

        # PCC
        pcc, _ = pearsonr(error, tv)
        pcc = round(pcc, 3)

        # Plot
        axes[idx].scatter(error, tv, s=8, alpha=0.6, c='blue', edgecolors='none')
        axes[idx].set_xlabel(f'{info["name"]} Error')
        axes[idx].set_ylabel(f'TV {head.capitalize()}')
        axes[idx].set_title(f'{info["name"]} Error vs Uncertainty\nPCC = {pcc}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "error_vs_uncertainty_combined_grid.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ====================== 5. Calibration Analysis - ECE + Reliability Plots ======================
    print("\n=== Uncertainty Calibration Analysis (ECE for Regression) ===")

    head_info = {
        'center': {'gt_key': 'center_gt_scale', 'channels': 2, 'name': 'Center'},
        'height': {'gt_key': 'height_gt_scale', 'channels': 1, 'name': 'Height'},
        'dim':    {'gt_key': 'dim_gt_scale',    'channels': 3, 'name': 'Dimension'},
        'rot':    {'gt_key': 'rot_gt_deg_scale', 'channels': 2, 'name': 'Rotation'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    num_bins = 15
    ece_scores = {}

    for idx, (head, info) in enumerate(head_info.items()):
        scale_key = f"{head}_scale"
        gt_key = info['gt_key']

        if scale_key not in unc or gt_key not in unc:
            axes[idx].text(0.5, 0.5, f'Missing {head}', ha='center', va='center')
            continue

        # Predicted values & uncertainty
        pred_values = unc[scale_key].numpy().reshape(-1, info['channels'])
        tv = compute_tv(unc[scale_key], channels=info['channels'])

        # GT values
        gt_values = unc[gt_key].numpy()
        if len(gt_values.shape) == 1:
            gt_values = gt_values.reshape(-1, 1)

        if head == 'center':
            gt_values = gt_values[:, 0:2]          # x,y only
        elif head == 'height' and gt_values.shape[1] >= 3:
            gt_values = gt_values[:, 2:3]          # z only

        # Align lengths
        min_len = min(len(pred_values), len(gt_values))
        pred_values = pred_values[:min_len]
        gt_values   = gt_values[:min_len]
        tv         = tv[:min_len]

        # Compute absolute error
        if head == 'rot':
            theta_pred = unc['rot_angle_deg_scale'].numpy()[:min_len].reshape(-1, 1)
            error = np.abs(theta_pred.flatten() - gt_values.flatten())
        else:
            error = np.linalg.norm(pred_values - gt_values, axis=1)

        # === Binning for Calibration ===
        tv_bins = np.linspace(tv.min(), tv.max(), num_bins + 1)
        bin_indices = np.digitize(tv, tv_bins) - 1
        bin_centers = (tv_bins[:-1] + tv_bins[1:]) / 2

        mean_pred_tv = []
        mean_obs_error = []

        for b in range(num_bins):
            mask = (bin_indices == b)
            if np.any(mask):
                mean_pred_tv.append(tv[mask].mean())
                mean_obs_error.append(error[mask].mean())

        mean_pred_tv = np.array(mean_pred_tv)
        mean_obs_error = np.array(mean_obs_error)

        # ECE (Expected Calibration Error)
        ece = np.mean(np.abs(mean_pred_tv - mean_obs_error))
        ece_scores[head] = round(ece, 4)

        # Plot
        axes[idx].scatter(mean_pred_tv, mean_obs_error, s=30, alpha=0.8, c='blue', edgecolors='none')
        axes[idx].plot([0, max(mean_pred_tv.max(), mean_obs_error.max())],
                       [0, max(mean_pred_tv.max(), mean_obs_error.max())], 'r--', label='Perfect Calibration')
        axes[idx].set_xlabel(f'Predicted Uncertainty (TV_{head})')
        axes[idx].set_ylabel(f'Observed Error')
        axes[idx].set_title(f'{info["name"]} Calibration\nECE = {ece:.4f}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "calibration_ece_combined_grid.png", dpi=400, bbox_inches='tight')
    plt.show()

    # Print ECE summary
    print("\nECE Scores (lower is better):")
    for head, ece in ece_scores.items():
        print(f"   {head:8s} → ECE = {ece}")


    print(f"\n🎉 All plots saved to: {save_dir}/")
    print("   → all_pairs_grid.png")
    print("   → tv_histograms.png")
    print("   → orientation_polar.png")
    print("   → orientation_vs_diff.png")
    print("   → aleatoric_uncertainty_vs_distance.png")
    print("   → rotation_error_vs_uncertainty.png")
    print("   → calibration_ece_combined_grid.png")


if __name__ == "__main__":
    main()