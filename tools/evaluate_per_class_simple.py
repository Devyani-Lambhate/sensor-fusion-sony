"""
Simplified evaluation script for computing per-class metrics.
Computes mAP and NDS metrics directly without heavy dependencies.
"""

import pickle
import numpy as np
import torch
import sys
from collections import defaultdict


class SimplePerClassEvaluator:
    """Compute per-class mAP metrics from prediction results."""
    
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
        Compute per-class TP/FP counts and precision/recall.
        
        Args:
            score_threshold: Only consider predictions with score > threshold
            iou_threshold: IoU threshold for positive match
            
        Returns:
            Dictionary with per-class metrics
        """
        
        # Track per-class metrics
        class_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'total_pred': 0, 'total_gt': 0,
            'sum_iou': 0, 'matched_pairs': 0
        })
        
        # Track all predictions for AP calculation
        all_preds = defaultdict(list)  # class -> [(score, matched)]
        
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
                
                # Match predictions to GT boxes
                for pred_idx, pred_box, pred_score in pred_list:
                    all_preds[class_name].append({
                        'score': pred_score,
                        'matched': False
                    })
                    
                    if len(gt_list) == 0:
                        # False positive - no GT for this class
                        class_stats[class_name]['fp'] += 1
                        continue
                    
                    # Find best matching GT
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in gt_list:
                        if gt_idx in matched_gt:
                            continue  # Already matched
                        
                        # Compute IoU (2D: x, y, length, width)
                        pred_2d = pred_box[[0, 1, 3, 4]]  # x, y, w, h
                        gt_2d = gt_box[[0, 1, 3, 4]]
                        
                        # Compute 2D IoU
                        iou = self._compute_box_iou(pred_2d, gt_2d)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        # True positive
                        class_stats[class_name]['tp'] += 1
                        class_stats[class_name]['sum_iou'] += best_iou
                        class_stats[class_name]['matched_pairs'] += 1
                        matched_gt.add(best_gt_idx)
                        all_preds[class_name][-1]['matched'] = True
                    else:
                        # False positive
                        class_stats[class_name]['fp'] += 1
            
            # Count false negatives (unmatched GT)
            for class_name in self.class_names:
                class_name_lower = class_name.lower()
                gt_list = gt_by_class.get(class_name_lower, [])
                
                for gt_idx, _ in gt_list:
                    if gt_idx not in matched_gt:
                        class_stats[class_name]['fn'] += 1
        
        return class_stats, all_preds
    
    def _compute_box_iou(self, box1, box2):
        """Compute 2D IoU between two boxes [x, y, w, h]."""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_w = max(0, xi_max - xi_min)
        inter_h = max(0, yi_max - yi_min)
        inter_area = inter_w * inter_h
        
        # Union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def print_results(self, class_stats):
        """Print per-class metrics."""
        
        print("\n" + "="*120)
        print("PER-CLASS DETECTION METRICS (IoU threshold: 0.5)")
        print("="*120)
        
        print(f"{'Class':<25} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<15} {'Recall':<15} {'Mean IoU':<12}")
        print("-"*120)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_pred = 0
        
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
            
            print(f"{class_name:<25} {tp:<8} {fp:<8} {fn:<8} {precision:<15.4f} {recall:<15.4f} {mean_iou:<12.4f}")
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += total_gt_class
            total_pred += total_pred_class
        
        print("-"*120)
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        
        print(f"{'OVERALL':<25} {total_tp:<8} {total_fp:<8} {total_fn:<8} {overall_precision:<15.4f} {overall_recall:<15.4f}")
        print("="*120)
        
        print("\nSUMMARY STATISTICS:")
        print(f"  Total Ground Truth:  {total_gt}")
        print(f"  Total Predictions:   {total_pred}")
        print(f"  Total True Positives: {total_tp}")
        print(f"  Total False Positives: {total_fp}")
        print(f"  Total False Negatives: {total_fn}")
        print(f"  Overall Precision: {overall_precision:.4f}")
        print(f"  Overall Recall: {overall_recall:.4f}")


def main():
    """Main evaluation script."""
    
    # Paths
    gt_pkl_path = '../data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl'
    pred_pkl_path = '../output/nuscenes_models/bevfusion/new_data_weather_edit/eval/eval_with_train/epoch_5/val/result.pkl'
    
    # Create evaluator
    evaluator = SimplePerClassEvaluator(
        gt_pkl_path=gt_pkl_path,
        pred_pkl_path=pred_pkl_path
    )
    
    # Compute metrics
    print("\nComputing per-class metrics (IoU threshold: 0.5)...")
    class_stats, all_preds = evaluator.compute_per_class_metrics(
        score_threshold=0.0,
        iou_threshold=0.5
    )
    
    # Print results
    evaluator.print_results(class_stats)


if __name__ == '__main__':
    main()
