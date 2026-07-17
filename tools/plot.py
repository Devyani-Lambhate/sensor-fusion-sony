import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log_for_iterations(log_file_path):
    iterations = []
    losses = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract Acc_iter (iteration number)
        iter_match = re.search(r'Acc_iter\s+(\d+)', line)
        if not iter_match:
            continue
            
        acc_iter = int(iter_match.group(1))
        
        # Extract smoothed loss from parentheses: (1.91e+03)
        loss_match = re.search(r'Loss:[\s\d\.]+\s*\(\s*([\d\.eE+-]+)', line)
        if loss_match:
            try:
                loss_val = float(loss_match.group(1))
                iterations.append(acc_iter)
                losses.append(loss_val)
            except:
                continue
    
    return iterations, losses


def plot_loss_vs_iteration(log_file_path, save_path="loss_vs_iteration.png"):
    iterations, losses = parse_training_log_for_iterations(log_file_path)
    
    if not iterations:
        print("❌ No iteration loss data found!")
        return
    
    print(f"✅ Extracted {len(iterations)} data points")
    print(f"First few: Iter {iterations[0]} → Loss {losses[0]:.1f}")
    print(f"Last few:  Iter {iterations[-1]} → Loss {losses[-1]:.1f}")
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, losses, linestyle='-', linewidth=1.8, color='blue', alpha=0.85, label='Smoothed Training Loss')
    
    # Optional: Add moving average for smoother trend
    window = 20
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(iterations[window-1:], moving_avg, 
                 linestyle='-', linewidth=2.5, color='red', label=f'{window}-point Moving Average')
    
    plt.title('Training Loss vs Iteration', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration (Acc_iter)', fontsize=14)
    plt.ylabel('Smoothed Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Mark epoch boundaries (optional but useful)
    plt.axvline(x=815, color='gray', linestyle='--', alpha=0.6, label='Epoch Boundary')
    plt.axvline(x=1630, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(x=2445, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(x=3260, color='gray', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved as: {save_path}")
    
    plt.show()


# ====================== RUN ======================
if __name__ == "__main__":
    log_file = "/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet_vishal/output/nuscenes_models/bevfusion/fog_finetuned_mixed/train_20260505-154146.log"
    
    plot_loss_vs_iteration(log_file)