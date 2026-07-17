import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet') 
from pcdet.utils.box_utils import boxes_iou_normal
import torch 

def plot_bev_boxes(boxes, color='r', label='pred'):
    for box in boxes:
        x, y, z, dx, dy, dz, yaw, vx, vy = box
        # Compute corners in BEV
        corners = np.array([
            [dx/2, dy/2],
            [dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2, dy/2]
        ])
        # Rotation
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        bev_corners = (corners @ rot.T) + np.array([x, y])
        plt.plot(*np.append(bev_corners, [bev_corners[0]], axis=0).T, color=color, label=label if label else None)
        label = None  # Only label first for legend

def compute_iou_metrics(pred_infos, gt_infos, score_threshold=0.1):
    total_tp = 0
    total_fp = 0
    total_gt = 0
    iou_list = []

    size_p = []
    iou_p = [] 
    dist_p = []
    
    # Track TPs per distance interval and per class
    distance_intervals = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    
    # Dictionary to store results: {class_name: {interval: {'tp': count, 'pred': count}}}
    tp_per_class_interval = {}
    pred_per_class_interval = {}
    
    for sample_idx in range(len(pred_infos)):
        gt_sample = gt_infos[sample_idx]
        gt_boxes = gt_sample['gt_boxes']
        gt_names = gt_sample.get('gt_names', [])
        total_gt += len(gt_boxes)
        
        pred_sample = pred_infos[sample_idx]
        indices = [i for i, val in enumerate(pred_sample['score']) if val > score_threshold]
        pred_boxes = pred_sample['boxes_lidar'][indices]
        pred_labels = pred_sample.get('pred_labels', [None] * len(pred_boxes))
        pred_labels = [pred_labels[i] for i in indices]
        
        # Use 'name' field for class names (this contains the actual class strings)
        pred_names = pred_sample.get('name', ['Unknown'] * len(pred_boxes))
        pred_names = [pred_names[i] for i in indices]
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
    
            pred_tensor = torch.from_numpy(pred_boxes[:, [0,1,3,4]]).float()
            gt_tensor = torch.from_numpy(gt_boxes[:, [0,1,3,4]]).float()
            ious = boxes_iou_normal(pred_tensor, gt_tensor).numpy()
            

            for i in range(len(pred_boxes)):
                # find the size of pred_boxes[i] in x and y dimensions
                dx = pred_boxes[i, 3]
                dy = pred_boxes[i, 4]
                dist  = np.sqrt(pred_boxes[i, 0]**2 + pred_boxes[i, 1]**2)
                
                # Get prediction class name directly from 'name' field
                pred_class_name = str(pred_names[i]).strip().lower() if i < len(pred_names) else "unknown"
                
                # Find best IoU with same-class GT boxes only
                best_iou_same_class = 0
                best_gt_idx = -1
                for j in range(len(gt_boxes)):
                    gt_class_name = str(gt_names[j]).strip().lower() if j < len(gt_names) else "unknown"
                    
                    # Match if classes are the same
                    if gt_class_name == pred_class_name and ious[i, j] > best_iou_same_class:
                        best_iou_same_class = ious[i, j]
                        best_gt_idx = j
                
                max_iou = best_iou_same_class
                if(max_iou == 0) : 
                    size_p.append(dx* dy)
                    iou_p.append(max_iou)
                    dist_p.append(dist)

                iou_list.append(max_iou)
                if max_iou > 0.5:
                    total_tp += 1
                else:
                    total_fp += 1
                
                # Track TP/predictions per distance interval and per class
                for (start, end) in distance_intervals:
                    if start <= dist < end:
                        interval = (start, end)
                        
                        # Initialize if not exists
                        if pred_class_name not in tp_per_class_interval:
                            tp_per_class_interval[pred_class_name] = {}
                            pred_per_class_interval[pred_class_name] = {}
                        if interval not in tp_per_class_interval[pred_class_name]:
                            tp_per_class_interval[pred_class_name][interval] = 0
                            pred_per_class_interval[pred_class_name][interval] = 0
                        
                        pred_per_class_interval[pred_class_name][interval] += 1
                        if max_iou > 0.5:
                            tp_per_class_interval[pred_class_name][interval] += 1
                        break
        
        elif len(pred_boxes) > 0:
            total_fp += len(pred_boxes)
            iou_list.extend([0.0] * len(pred_boxes))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    mean_iou = np.mean(iou_list) if iou_list else 0
    plt.hist(dist_p, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')
    return {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_gt': total_gt,
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou,
        'tp_per_class_interval': tp_per_class_interval,
        'pred_per_class_interval': pred_per_class_interval,
        'distance_intervals': distance_intervals
    }

def main():
    pred_pkl_path = '../output/nuscenes_models/bevfusion/new_data_weather_edit/eval/eval_with_train/epoch_5/val/result.pkl'
    gt_pkl_path = '../data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl'  # Update path if needed
    sample_idx = 10  # Change index as needed

    # Load ground truth
    with open(gt_pkl_path, 'rb') as f:
        gt_infos = pickle.load(f)

    with open(pred_pkl_path, 'rb') as f:
        pred_infos = pickle.load(f)

    # Compute IOU metrics for all samples
    metrics = compute_iou_metrics(pred_infos, gt_infos, score_threshold=0.1)
    print("IOU Metrics across all samples:")
    print(f"Total True Positives: {metrics['total_tp']}")
    print(f"Total False Positives: {metrics['total_fp']}")
    print(f"Total Ground Truth: {metrics['total_gt']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Mean IOU: {metrics['mean_iou']:.3f}")
    
    # Print per-class and per-distance interval results
    print("\n" + "="*80)
    print("Correctly Predicted Boxes by Class and Distance Interval:")
    print("="*80)
    
    class_wise_stats = {}
    
    for class_label in sorted(metrics['tp_per_class_interval'].keys()):
        print(f"\nClass: {class_label}")
        print("-" * 80)
        print(f"{'Distance Interval':<20} {'TP / Total':<15} {'Accuracy':<15}")
        print("-" * 80)
        
        class_total_tp = 0
        class_total_pred = 0
        
        for (start, end) in sorted(metrics['distance_intervals']):
            interval = (start, end)
            tp = metrics['tp_per_class_interval'].get(class_label, {}).get(interval, 0)
            total_pred = metrics['pred_per_class_interval'].get(class_label, {}).get(interval, 0)
            
            class_total_tp += tp
            class_total_pred += total_pred
            
            if total_pred > 0:
                accuracy = tp / total_pred
                print(f"{start}-{end}m{'':<12} {tp}/{total_pred:<13} {accuracy:.2%}")
            else:
                print(f"{start}-{end}m{'':<12} {'No predictions':<13} {'-':>14}")
        
        if class_total_pred > 0:
            class_wise_stats[class_label] = {
                'total_tp': class_total_tp,
                'total_pred': class_total_pred,
                'accuracy': class_total_tp / class_total_pred
            }
    
    # Print class-wise summary
    print("\n" + "="*80)
    print("Class-wise Summary (all distance ranges):")
    print("="*80)
    print(f"{'Class':<25} {'TP / Total':<15} {'Accuracy':<15}")
    print("-" * 80)
    for class_label in sorted(class_wise_stats.keys()):
        stats = class_wise_stats[class_label]
        print(f"{class_label:<25} {stats['total_tp']}/{stats['total_pred']:<13} {stats['accuracy']:.2%}")
    print("="*80)
    
    gt_sample = gt_infos[sample_idx]
    gt_boxes = gt_sample['gt_boxes']

    pred_sample = pred_infos[sample_idx]
    indices = [i for i, val in enumerate(pred_sample['score']) if val > 0.1]
    pred_boxes = pred_sample['boxes_lidar'][indices]

    plt.figure(figsize=(10, 10))
    plot_bev_boxes(pred_boxes, color='r', label='Prediction')
    plot_bev_boxes(gt_boxes, color='b', label='Ground Truth')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Predicted and Ground Truth Boxes in BEV')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()