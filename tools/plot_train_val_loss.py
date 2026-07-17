import argparse
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def parse_tensorboard_events(tensorboard_dir):
    """
    Extract loss data from tensorboard event files.
    
    Args:
        tensorboard_dir: Path to tensorboard directory containing events
        
    Returns:
        train_losses, val_losses, steps
    """
    if not HAS_TENSORBOARD:
        print("Warning: tensorboard not installed. Install with: pip install tensorboard")
        return [], [], []
    
    try:
        ea = EventAccumulator(str(tensorboard_dir))
        ea.Reload()
        
        train_losses = []
        val_losses = []
        steps = []
        
        # Get all available scalars
        scalars = ea.Tags().get('scalars', [])
        print(f"  Available metrics: {scalars}")
        
        # Extract training and validation losses
        for scalar_name in scalars:
            try:
                events = ea.Scalars(scalar_name)
                
                # Categorize losses
                is_train = 'train' in scalar_name.lower()
                is_val = 'val' in scalar_name.lower() or 'evaluation' in scalar_name.lower()
                is_loss = 'loss' in scalar_name.lower()
                
                if is_loss:
                    for event in events:
                        step = event.step
                        value = event.value
                        
                        if is_val:
                            if step not in [s for s, _ in val_losses]:
                                val_losses.append((step, value))
                        else:  # Default to training if not explicitly validation
                            if step not in steps:
                                steps.append(step)
                            train_losses.append((step, value))
            except:
                continue
        
        # Sort by step
        if train_losses:
            train_losses.sort(key=lambda x: x[0])
            steps = sorted(list(set(steps)))
            train_losses = [v for _, v in train_losses]
        
        if val_losses:
            val_losses.sort(key=lambda x: x[0])
            val_losses = [v for _, v in val_losses]
        
        return train_losses, val_losses, steps
    except Exception as e:
        print(f"Error reading tensorboard events: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []


def parse_log_file(log_file_path):
    """
    Parse training log file and extract losses.
    
    Expected log format:
    Train:    epoch/total (%) [batch/total (percent)]  Loss: value (avg_loss)  ...
    """
    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Parse training loss lines
            # Example: Train:    1/1 (100%) [ 799/815 ( 98%)]  Loss: 1.723 (2.52)  LR: ...
            train_match = re.search(r'Train:\s+(\d+)/\d+.*Loss:\s+([\d.]+)\s+\(', line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                if epoch not in epochs:
                    epochs.append(epoch)
                train_losses.append(loss)
            
            # Parse validation loss lines (typically logged after each epoch)
            # Example: Epoch: 1, total_loss: 1.234, ...
            val_match = re.search(r'Epoch:\s+(\d+).*total_loss:\s+([\d.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(2)))
    
    return train_losses, val_losses, sorted(list(set(epochs)))


def plot_train_val_loss(experiment_dir=None, log_file_path=None, tensorboard_dir=None, output_dir=None, save_name='train_val_loss.png'):
    """
    Plot training and validation loss from tensorboard events or log file.
    
    Args:
        experiment_dir: Path to experiment directory (auto-detects tensorboard/log)
        log_file_path: Path to specific training log file
        tensorboard_dir: Path to tensorboard directory with events
        output_dir: Directory to save plot (defaults to experiment_dir or same as log)
        save_name: Name of output image file
    """
    train_losses = []
    val_losses = []
    
    # Auto-detect tensorboard directory
    if experiment_dir:
        experiment_dir = Path(experiment_dir)
        tb_dir = experiment_dir / 'tensorboard'
        if tb_dir.exists() and tensorboard_dir is None:
            tensorboard_dir = tb_dir
        
        # Look for log files
        if log_file_path is None:
            log_files = list(experiment_dir.glob('train_*.log'))
            if log_files:
                log_file_path = log_files[0]
        
        if output_dir is None:
            output_dir = experiment_dir
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from tensorboard first (has more detailed data)
    if tensorboard_dir and HAS_TENSORBOARD:
        print(f"Loading from tensorboard events: {tensorboard_dir}")
        train_losses, val_losses, steps = parse_tensorboard_events(str(tensorboard_dir))
        if not train_losses:
            print("No losses found in tensorboard events, trying log file...")
        else:
            print(f"Loaded {len(train_losses)} training loss values from tensorboard")
    
    # Fallback to log file
    if not train_losses and log_file_path:
        if not os.path.exists(log_file_path):
            print(f"Error: Log file not found at {log_file_path}")
            return False
        
        print(f"Parsing log file: {log_file_path}")
        train_losses, val_losses, epochs = parse_log_file(str(log_file_path))
    
    if not train_losses:
        print("No training losses found. Please provide a valid tensorboard directory or log file.")
        return False
    
    # Create figure
    if val_losses:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    
    # Plot training loss
    iterations = np.arange(len(train_losses))
    axes[0].plot(iterations, train_losses, label='Training Loss', linewidth=2, marker='o', markersize=3)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Plot validation loss if available
    if val_losses and len(axes) > 1:
        epochs = np.arange(1, len(val_losses) + 1)
        axes[1].plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=6, color='orange')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_path = output_dir / save_name
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    # Display statistics
    print(f"\n{'='*50}")
    print(f"Training Statistics:")
    print(f"  Total iterations: {len(train_losses)}")
    print(f"  Min loss: {min(train_losses):.4f}")
    print(f"  Max loss: {max(train_losses):.4f}")
    print(f"  Mean loss: {np.mean(train_losses):.4f}")
    print(f"  Final loss: {train_losses[-1]:.4f}")
    
    if val_losses:
        print(f"\nValidation Statistics:")
        print(f"  Total epochs: {len(val_losses)}")
        print(f"  Min loss: {min(val_losses):.4f}")
        print(f"  Max loss: {max(val_losses):.4f}")
        print(f"  Mean loss: {np.mean(val_losses):.4f}")
        print(f"  Final loss: {val_losses[-1]:.4f}")
    print(f"{'='*50}\n")
    
    plt.show()
    return True


def plot_experiment_loss(experiment_dir, output_dir=None):
    """
    Plot loss for an experiment using tensorboard or log files.
    
    Args:
        experiment_dir: Path to experiment directory containing log files or tensorboard
        output_dir: Directory to save plots (defaults to experiment_dir)
    """
    experiment_dir = Path(experiment_dir)
    if output_dir is None:
        output_dir = experiment_dir
    else:
        output_dir = Path(output_dir)
    
    print(f"Analyzing experiment in: {experiment_dir}")
    return plot_train_val_loss(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        save_name='train_val_loss.png'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot training and validation loss from experiment logs or tensorboard events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using experiment directory (auto-detects tensorboard/logs)
  python plot_train_val_loss.py --experiment_dir /path/to/experiment
  
  # Using specific log file
  python plot_train_val_loss.py --log_file /path/to/train_*.log
  
  # Using tensorboard directory
  python plot_train_val_loss.py --tensorboard_dir /path/to/tensorboard
  
  # Custom output directory
  python plot_train_val_loss.py --experiment_dir /path/to/exp --output_dir /path/to/output
  
  # Show all available metrics without plotting
  python plot_train_val_loss.py --experiment_dir /path/to/exp --list_metrics
        """
    )
    parser.add_argument('--experiment_dir', type=str, default=None, 
                        help='Path to experiment directory (auto-detects tensorboard/logs)')
    parser.add_argument('--log_file', type=str, default=None, 
                        help='Path to specific training log file')
    parser.add_argument('--tensorboard_dir', type=str, default=None, 
                        help='Path to tensorboard events directory')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for plots')
    parser.add_argument('--save_name', type=str, default='train_val_loss.png', 
                        help='Output filename')
    parser.add_argument('--list_metrics', action='store_true',
                        help='List available metrics in tensorboard without plotting')
    
    args = parser.parse_args()
    
    if args.list_metrics and args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        tb_dir = experiment_dir / 'tensorboard'
        if tb_dir.exists() and HAS_TENSORBOARD:
            try:
                ea = EventAccumulator(str(tb_dir))
                ea.Reload()
                scalars = ea.Tags().get('scalars', [])
                print(f"\nAvailable metrics in {tb_dir}:")
                for metric in sorted(scalars):
                    print(f"  - {metric}")
            except Exception as e:
                print(f"Error reading tensorboard: {e}")
        else:
            print(f"Tensorboard directory not found at {tb_dir}")
    elif args.experiment_dir:
        plot_experiment_loss(args.experiment_dir, args.output_dir)
    elif args.log_file or args.tensorboard_dir:
        plot_train_val_loss(
            log_file_path=args.log_file,
            tensorboard_dir=args.tensorboard_dir,
            output_dir=args.output_dir,
            save_name=args.save_name
        )
    else:
        parser.print_help()
