"""
Enhanced per-class evaluation script with mAP@0.5 computation.
Computes detailed metrics for each class including precision, recall, and average precision.
"""

import pickle
import numpy as np
from collections import defaultdict
import csv
from pathlib import Path


class EnhancedPerClassEvaluator:
    """Compute per-class metrics including mAP from prediction results."""
    
    def __init__(self, gt_pkl_path, pred_pkl_path):
        """
        Args:
            gt_pkl_path: Path to ground truth info pickle
            pred_pkl_path: Path to prediction results pickle
        """
        self.gt_pkl_path = gt_pkl_path
        self.pred_pkl_path = pred_pkl_path
        
        # Load pickle files
        with open(gt_pkl_path, 'rb') as f:
            self.gt_infos = pickle.load(f)
        
        with open(pred_pkl_path, 'rb') as f:
            self.pred_infos = pickle.load(f)
        
        # Class names
        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        print(f"Loaded {len(self.gt_infos)} GT samples and {len(self.pred_infos)} prediction samples")
    
    def compute_per_class_metrics(self, score_threshold=0.0, iou_threshold=0.5):
        """
        Compute per-class metrics and average precision.
        
        Args:
            score_threshold: Only consider predictions with score > threshold
            iou_threshold: IoU threshold for positive match
            
        Returns:
            Dictionary with per-class metrics
        """
        
        # Track per-class predictions for AP calculation
        class_predictions = defaultdict(list)  # class -> [(score, matched)]
        
        # Track per-class statistics
        class_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'total_pred': 0, 'total_gt': 0,
            'sum_iou': 0, 'matched_pairs': 0
        })
        
        print("\nProcessing samples...")
        for sample_idx in range(len(self.gt_infos)):
            if sample_idx % 10 == 0:
                print(f"  Sample {sample_idx}/{len(self.gt_infos)}")
            
            gt_sample = self.gt_infos[sample_idx]
            pred_sample = self.pred_infos[sample_idx]
            
            gt_boxes = gt_sample['gt_boxes']
            gt_names = gt_sample.get('gt_names', [])
            
            # Filter predictions by score
            scores = np.array(pred_sample['score'])
            score_mask = scores > score_threshold
            
            pred_boxes = pred_sample['boxes_lidar'][score_mask]
            pred_scores = scores[score_mask]
            pred_names = np.array(pred_sample['name'])[score_mask]
            
            # Group by class
            gt_by_class = defaultdict(list)
            for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                class_name = str(name).lower().strip()
                gt_by_class[class_name].append((i, box))
            
            pred_by_class = defaultdict(list)
            for i, (box, score, name) in enumerate(zip(pred_boxes, pred_scores, pred_names)):
                class_name = str(name).lower().strip()
                pred_by_class[class_name].append((i, box, score))
            
            # Match predictions to GT for each class
            matched_gt = set()
            
            for class_name in self.class_names:
                class_name_lower = class_name.lower()
                
                # Get GT and predictions for this class
                gt_list = gt_by_class.get(class_name_lower, [])
                pred_list = pred_by_class.get(class_name_lower, [])
                
                class_stats[class_name]['total_gt'] += len(gt_list)
                class_stats[class_name]['total_pred'] += len(pred_list)
                
                # Sort predictions by score (descending) for AP calculation
                pred_list_sorted = sorted(pred_list, key=lambda x: x[2], reverse=True)
                
                # Match predictions to GT boxes
                for pred_idx, pred_box, pred_score in pred_list_sorted:
                    is_matched = False
                    
                    if len(gt_list) > 0:
                        # Find best matching GT
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx, gt_box in gt_list:
                            if gt_idx in matched_gt:
                                continue  # Already matched
                            
                            # Compute 2D IoU
                            iou = self._compute_box_iou(pred_box, gt_box)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        
                        if best_iou >= iou_threshold and best_gt_idx >= 0:
                            # True positive
                            is_matched = True
                            class_stats[class_name]['tp'] += 1
                            class_stats[class_name]['sum_iou'] += best_iou
                            class_stats[class_name]['matched_pairs'] += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            # False positive
                            class_stats[class_name]['fp'] += 1
                    else:
                        # False positive - no GT for this class
                        class_stats[class_name]['fp'] += 1
                    
                    # Store for AP calculation
                    class_predictions[class_name].append({
                        'score': pred_score,
                        'matched': is_matched
                    })
            
            # Count false negatives
            for class_name in self.class_names:
                class_name_lower = class_name.lower()
                gt_list = gt_by_class.get(class_name_lower, [])
                
                for gt_idx, _ in gt_list:
                    if gt_idx not in matched_gt:
                        class_stats[class_name]['fn'] += 1
        
        # Compute AP for each class
        ap_dict = self._compute_ap_per_class(class_predictions)
        
        return class_stats, ap_dict
    
    def _compute_box_iou(self, box1, box2):
        """Compute 2D IoU between two boxes [x, y, z, dx, dy, dz, yaw, ...]."""
        # Extract 2D coordinates and dimensions
        x1_min = box1[0] - box1[3] / 2
        y1_min = box1[1] - box1[4] / 2
        x1_max = box1[0] + box1[3] / 2
        y1_max = box1[1] + box1[4] / 2
        
        x2_min = box2[0] - box2[3] / 2
        y2_min = box2[1] - box2[4] / 2
        x2_max = box2[0] + box2[3] / 2
        y2_max = box2[1] + box2[4] / 2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_w = max(0, xi_max - xi_min)
        inter_h = max(0, yi_max - yi_min)
        inter_area = inter_w * inter_h
        
        # Union
        box1_area = box1[3] * box1[4]
        box2_area = box2[3] * box2[4]
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def _compute_ap_per_class(self, class_predictions, num_recall_samples=11):
        """Compute Average Precision for each class."""
        ap_dict = {}
        
        for class_name in self.class_names:
            if class_name not in class_predictions or len(class_predictions[class_name]) == 0:
                ap_dict[class_name] = 0.0
                continue
            
            # Sort predictions by score
            predictions = sorted(
                class_predictions[class_name],
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Compute precision and recall
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            total_gt = sum(1 for p in class_predictions[class_name] if p['matched'] or not p['matched'])
            total_gt = max(1, sum(p['matched'] for p in class_predictions[class_name]))
            
            for pred in predictions:
                if pred['matched']:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / max(total_gt, 1)
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP using all points
            if len(precisions) > 0:
                ap = np.trapz(precisions, recalls) if len(precisions) > 1 else np.mean(precisions)
            else:
                ap = 0.0
            
            ap_dict[class_name] = max(0, ap)  # Ensure non-negative
        
        return ap_dict
    
    def print_results(self, class_stats, ap_dict):
        """Print per-class metrics in a formatted table."""
        
        print("\n" + "="*140)
        print("PER-CLASS EVALUATION RESULTS")
        print("="*140)
        
        print(f"{'Class':<25} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<15} {'Recall':<15} {'Mean IoU':<12} {'mAP@0.5':<12}")
        print("-"*140)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_pred = 0
        total_aps = 0
        num_classes = 0
        
        for class_name in self.class_names:
            stats = class_stats.get(class_name, {})
            
            tp = stats.get('tp', 0)
            fp = stats.get('fp', 0)
            fn = stats.get('fn', 0)
            total_gt_class = stats.get('total_gt', 0)
            total_pred_class = stats.get('total_pred', 0)
            
            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_gt_class if total_gt_class > 0 else 0
            mean_iou = stats.get('sum_iou', 0) / max(stats.get('matched_pairs', 0), 1)
            ap = ap_dict.get(class_name, 0)
            
            # Print row
            print(f"{class_name:<25} {tp:<8} {fp:<8} {fn:<8} {precision:<15.4f} {recall:<15.4f} {mean_iou:<12.4f} {ap:<12.4f}")
            
            # Accumulate totals
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += total_gt_class
            total_pred += total_pred_class
            if total_gt_class > 0:  # Only count classes with GT
                total_aps += ap
                num_classes += 1
        
        print("-"*140)
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        mean_ap = total_aps / max(num_classes, 1)
        
        print(f"{'OVERALL':<25} {total_tp:<8} {total_fp:<8} {total_fn:<8} {overall_precision:<15.4f} {overall_recall:<15.4f} {'-':<12} {mean_ap:<12.4f}")
        print("="*140)
        
        return {
            'mAP': mean_ap,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def save_to_csv(self, filename, class_stats, ap_dict):
        """Save results to CSV file."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'Mean IoU', 'mAP@0.5'])
            
            for class_name in self.class_names:
                stats = class_stats.get(class_name, {})
                tp = stats.get('tp', 0)
                fp = stats.get('fp', 0)
                fn = stats.get('fn', 0)
                total_gt = stats.get('total_gt', 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0
                mean_iou = stats.get('sum_iou', 0) / max(stats.get('matched_pairs', 0), 1)
                ap = ap_dict.get(class_name, 0)
                
                writer.writerow([
                    class_name,
                    tp, fp, fn,
                    f'{precision:.4f}',
                    f'{recall:.4f}',
                    f'{mean_iou:.4f}',
                    f'{ap:.4f}'
                ])
        
        print(f"\nResults saved to {output_path}")


def main():
    """Main evaluation script."""
    
    # Paths
    gt_pkl_path = '../data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl'
    pred_pkl_path = '../output/nuscenes_models/bevfusion/new_data_weather_edit/eval/eval_with_train/epoch_5/val/result.pkl'
    output_csv = 'evaluation_results.csv'
    
    # Create evaluator
    evaluator = EnhancedPerClassEvaluator(
        gt_pkl_path=gt_pkl_path,
        pred_pkl_path=pred_pkl_path
    )
    
    # Compute metrics
    print("\nComputing per-class metrics (IoU threshold: 0.5)...")
    class_stats, ap_dict = evaluator.compute_per_class_metrics(
        score_threshold=0.0,
        iou_threshold=0.5
    )
    
    # Print results
    summary = evaluator.print_results(class_stats, ap_dict)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Mean Average Precision (mAP@0.5): {summary['mAP']:.4f}")
    print(f"  Overall Precision: {summary['overall_precision']:.4f}")
    print(f"  Overall Recall: {summary['overall_recall']:.4f}")
    print(f"  Total TP: {summary['total_tp']}")
    print(f"  Total FP: {summary['total_fp']}")
    print(f"  Total FN: {summary['total_fn']}")
    
    # Save to CSV
    evaluator.save_to_csv(output_csv, class_stats, ap_dict)


if __name__ == '__main__':
    main()
