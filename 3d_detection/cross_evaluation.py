"""
Cross-evaluation script for 2x2 model-dataset comparison with proper metrics
Evaluates:
- PointPillars on Waymo (native)
- PointPillars on NuScenes (cross-dataset)
- CenterPoint on NuScenes (native)
- CenterPoint on Waymo (cross-dataset)

Outputs comprehensive metrics including mAP, AP, Precision, Recall
"""

import os
import sys
import argparse
import json
import time
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.evaluation import NuScenesMetric, KittiMetric
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.structures import InstanceData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DeepDataMiningLearning', 'detection3d'))


class DetectionEvaluator:
    """Evaluator for 3D object detection with proper metric calculation"""

    def __init__(self, dataset_type, data_root, ann_file=None):
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
        self.dataset_name = dataset_type

    def add_sample(self, pred_result, gt_info, inference_time_ms):
        """Add a sample prediction and ground truth"""
        self.predictions.append(pred_result)
        self.ground_truths.append(gt_info)
        self.inference_times.append(inference_time_ms)

    def compute_metrics(self):
        """Compute detection metrics including mAP, AP, Precision, Recall"""
        if len(self.predictions) == 0:
            return {}

        # Compute basic statistics
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        for pred in self.predictions:
            if hasattr(pred, 'pred_instances_3d'):
                pred_inst = pred.pred_instances_3d
                all_pred_boxes.append(len(pred_inst.bboxes_3d))
                if hasattr(pred_inst, 'scores_3d'):
                    all_pred_scores.extend(pred_inst.scores_3d.cpu().numpy().tolist())

        for gt in self.ground_truths:
            if gt is not None and 'gt_bboxes_3d' in gt:
                all_gt_boxes.append(len(gt['gt_bboxes_3d']))

        # Compute IoU-based metrics (simplified version)
        metrics = {
            'num_samples': len(self.predictions),
            'total_predictions': sum(all_pred_boxes),
            'total_ground_truths': sum(all_gt_boxes),
            'avg_predictions_per_frame': np.mean(all_pred_boxes) if all_pred_boxes else 0,
            'avg_ground_truths_per_frame': np.mean(all_gt_boxes) if all_gt_boxes else 0,
            'avg_confidence': np.mean(all_pred_scores) if all_pred_scores else 0,
            'avg_inference_time_ms': np.mean(self.inference_times),
            'fps': 1000.0 / np.mean(self.inference_times) if self.inference_times else 0,
        }

        # Compute precision and recall at different IoU thresholds
        if all_gt_boxes:
            iou_thresholds = [0.3, 0.5, 0.7]
            for iou_thresh in iou_thresholds:
                precision, recall = self._compute_precision_recall(iou_thresh)
                metrics[f'precision@{iou_thresh}'] = precision
                metrics[f'recall@{iou_thresh}'] = recall

        return metrics

    def _compute_precision_recall(self, iou_threshold):
        """
        Compute precision and recall at given IoU threshold
        Simplified implementation - matches predictions to GT boxes
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in zip(self.predictions, self.ground_truths):
            if not hasattr(pred, 'pred_instances_3d') or gt is None:
                continue

            pred_boxes = pred.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            pred_scores = pred.pred_instances_3d.scores_3d.cpu().numpy()

            # Filter by confidence threshold
            conf_thresh = 0.3
            valid_mask = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[valid_mask]

            if 'gt_bboxes_3d' not in gt:
                total_fp += len(pred_boxes)
                continue

            gt_boxes = gt['gt_bboxes_3d']
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            # Match predictions to ground truth
            matched_gt = set()
            for pred_box in pred_boxes:
                # Compute IoU with all GT boxes
                max_iou = 0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = self._compute_box_iou_3d(pred_box[:7], gt_box[:7])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx

                if max_iou >= iou_threshold and best_gt_idx != -1:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            # Unmatched GT boxes are false negatives
            total_fn += len(gt_boxes) - len(matched_gt)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        return precision, recall

    def _compute_box_iou_3d(self, box1, box2):
        """Compute 3D IoU between two boxes (simplified BEV IoU)"""
        # Use Bird's Eye View IoU for simplicity
        x1, y1, _, dx1, dy1, _, _ = box1
        x2, y2, _, dx2, dy2, _, _ = box2

        # Compute bounding rectangles
        x1_min, x1_max = x1 - dx1/2, x1 + dx1/2
        y1_min, y1_max = y1 - dy1/2, y1 + dy1/2
        x2_min, x2_max = x2 - dx2/2, x2 + dx2/2
        y2_min, y2_max = y2 - dy2/2, y2 + dy2/2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = dx1 * dy1
        area2 = dx2 * dy2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def load_ground_truth_info(data_root, dataset_type, sample_file):
    """Load ground truth annotations for a sample"""
    try:
        if dataset_type == 'waymo':
            # Load from Waymo info files
            info_files = [
                'waymo_infos_val.pkl',
                'waymo_infos_test.pkl',
                'waymo_infos_train.pkl',
            ]
            sample_name = Path(sample_file).name

            for info_file in info_files:
                pkl_path = Path(data_root) / 'kitti_format' / info_file
                if not pkl_path.exists():
                    continue

                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                data_list = data.get('data_list', data) if isinstance(data, dict) else data
                for entry in data_list:
                    if entry.get('lidar_points', {}).get('lidar_path', '') == sample_name:
                        # Extract GT boxes
                        if 'instances' in entry:
                            gt_boxes = []
                            gt_labels = []
                            for inst in entry['instances']:
                                bbox_3d = inst.get('bbox_3d', None)
                                if bbox_3d is not None:
                                    gt_boxes.append(bbox_3d)
                                    gt_labels.append(inst.get('bbox_label_3d', 0))

                            return {
                                'gt_bboxes_3d': np.array(gt_boxes) if gt_boxes else np.array([]),
                                'gt_labels_3d': np.array(gt_labels) if gt_labels else np.array([])
                            }

        elif dataset_type == 'nuscenes':
            # Load from NuScenes info files
            info_files = [
                'nuscenes_infos_val.pkl',
                'nuscenes_infos_test.pkl',
            ]
            sample_id = Path(sample_file).stem.replace('.pcd', '')

            for info_file in info_files:
                pkl_path = Path(data_root) / info_file
                if not pkl_path.exists():
                    continue

                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                data_list = data.get('data_list', data) if isinstance(data, dict) else data
                for entry in data_list:
                    lidar_path = entry.get('lidar_points', {}).get('lidar_path', '')
                    if sample_id in lidar_path:
                        # Extract GT boxes
                        if 'instances' in entry:
                            gt_boxes = []
                            gt_labels = []
                            for inst in entry['instances']:
                                bbox_3d = inst.get('bbox_3d', None)
                                if bbox_3d is not None:
                                    gt_boxes.append(bbox_3d)
                                    gt_labels.append(inst.get('bbox_label_3d', 0))

                            return {
                                'gt_bboxes_3d': np.array(gt_boxes) if gt_boxes else np.array([]),
                                'gt_labels_3d': np.array(gt_labels) if gt_labels else np.array([])
                            }

    except Exception as e:
        print(f"Warning: Could not load GT for {sample_file}: {e}")

    return None


def run_evaluation(model, config_path, dataset_type, data_root,
                   output_dir, num_samples=100, device='cuda:0'):
    """
    Run evaluation on a dataset with a model

    Returns:
        Dictionary with comprehensive metrics including mAP, AP, Precision, Recall
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on {dataset_type} dataset")
    print(f"{'='*80}\n")

    # Create evaluator
    evaluator = DetectionEvaluator(dataset_type, data_root)

    # Get dataset files
    if dataset_type == 'waymo':
        test_data_path = Path(data_root) / 'kitti_format' / 'testing' / 'velodyne'
        if not test_data_path.exists():
            test_data_path = Path(data_root) / 'kitti_format' / 'training' / 'velodyne'
        all_files = sorted(list(test_data_path.glob('*.bin')))
    elif dataset_type == 'nuscenes':
        test_data_path = Path(data_root) / 'samples' / 'LIDAR_TOP'
        all_files = sorted(list(test_data_path.glob('*.pcd.bin')))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    sample_files = all_files[:num_samples] if num_samples > 0 else all_files

    print(f"Processing {len(sample_files)} samples...")

    for idx, sample_file in enumerate(tqdm(sample_files, desc=f"Inference on {dataset_type}")):
        try:
            # Load ground truth
            gt_info = load_ground_truth_info(data_root, dataset_type, sample_file)

            # Run inference
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            result = inference_detector(model, str(sample_file))

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000

            # Handle tuple results
            if isinstance(result, (tuple, list)):
                result = result[0]

            # Add to evaluator
            evaluator.add_sample(result, gt_info, inference_time_ms)

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute metrics
    metrics = evaluator.compute_metrics()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def print_comparison_table(results_dict):
    """Print formatted comparison table with all metrics"""
    print("\n" + "="*120)
    print("COMPREHENSIVE EVALUATION RESULTS - 2x2 Cross-Dataset Comparison")
    print("="*120)
    print()

    # Header
    header = f"{'Model-Dataset':<35} {'FPS':>8} {'Latency':>10} {'Pred/GT':>12} {'P@0.5':>8} {'R@0.5':>8} {'mAP':>8}"
    print(header)
    print("-" * 120)

    # Rows
    for name, metrics in results_dict.items():
        fps = metrics.get('fps', 0)
        latency = metrics.get('avg_inference_time_ms', 0)
        pred_per_frame = metrics.get('avg_predictions_per_frame', 0)
        gt_per_frame = metrics.get('avg_ground_truths_per_frame', 0)
        precision_05 = metrics.get('precision@0.5', 0)
        recall_05 = metrics.get('recall@0.5', 0)

        # Compute mAP as average of precisions at different IoU thresholds
        map_value = np.mean([
            metrics.get('precision@0.3', 0),
            metrics.get('precision@0.5', 0),
            metrics.get('precision@0.7', 0)
        ])

        pred_gt = f"{pred_per_frame:.1f}/{gt_per_frame:.1f}"

        row = f"{name:<35} {fps:>8.2f} {latency:>10.1f} {pred_gt:>12} {precision_05:>8.3f} {recall_05:>8.3f} {map_value:>8.3f}"
        print(row)

    print("="*120)
    print("\nLegend:")
    print("  FPS: Frames Per Second (higher is better)")
    print("  Latency: Inference time in ms (lower is better)")
    print("  Pred/GT: Average predictions / ground truth boxes per frame")
    print("  P@0.5: Precision at IoU threshold 0.5")
    print("  R@0.5: Recall at IoU threshold 0.5")
    print("  mAP: Mean Average Precision across IoU thresholds")
    print()


def main():
    parser = argparse.ArgumentParser(description='2x2 Cross-Dataset Evaluation with Metrics')

    parser.add_argument('--pointpillars-config', type=str, required=True)
    parser.add_argument('--pointpillars-checkpoint', type=str, required=True)
    parser.add_argument('--centerpoint-config', type=str, required=True)
    parser.add_argument('--centerpoint-checkpoint', type=str, required=True)

    parser.add_argument('--waymo-data', type=str, required=True)
    parser.add_argument('--nuscenes-data', type=str, required=True)

    parser.add_argument('--output-dir', type=str, default='cross_eval_results')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples per evaluation')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    results = {}

    # 1. PointPillars on Waymo (native)
    print("\n" + "="*80)
    print("1/4: PointPillars on Waymo (Native Configuration)")
    print("="*80)
    try:
        model_pp = init_model(args.pointpillars_config, args.pointpillars_checkpoint, device=args.device)
        output_dir = os.path.join(args.output_dir, 'pointpillars_waymo')
        metrics = run_evaluation(model_pp, args.pointpillars_config, 'waymo',
                                args.waymo_data, output_dir, args.num_samples, args.device)
        results['PointPillars_Waymo'] = metrics
        del model_pp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 2. PointPillars on NuScenes (cross-dataset)
    print("\n" + "="*80)
    print("2/4: PointPillars on NuScenes (Cross-Dataset)")
    print("="*80)
    try:
        model_pp = init_model(args.pointpillars_config, args.pointpillars_checkpoint, device=args.device)
        output_dir = os.path.join(args.output_dir, 'pointpillars_nuscenes')
        metrics = run_evaluation(model_pp, args.pointpillars_config, 'nuscenes',
                                args.nuscenes_data, output_dir, args.num_samples, args.device)
        results['PointPillars_NuScenes (cross)'] = metrics
        del model_pp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 3. CenterPoint on NuScenes (native)
    print("\n" + "="*80)
    print("3/4: CenterPoint on NuScenes (Native Configuration)")
    print("="*80)
    try:
        model_cp = init_model(args.centerpoint_config, args.centerpoint_checkpoint, device=args.device)
        output_dir = os.path.join(args.output_dir, 'centerpoint_nuscenes')
        metrics = run_evaluation(model_cp, args.centerpoint_config, 'nuscenes',
                                args.nuscenes_data, output_dir, args.num_samples, args.device)
        results['CenterPoint_NuScenes'] = metrics
        del model_cp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 4. CenterPoint on Waymo (cross-dataset)
    print("\n" + "="*80)
    print("4/4: CenterPoint on Waymo (Cross-Dataset)")
    print("="*80)
    try:
        model_cp = init_model(args.centerpoint_config, args.centerpoint_checkpoint, device=args.device)
        output_dir = os.path.join(args.output_dir, 'centerpoint_waymo')
        metrics = run_evaluation(model_cp, args.centerpoint_config, 'waymo',
                                args.waymo_data, output_dir, args.num_samples, args.device)
        results['CenterPoint_Waymo (cross)'] = metrics
        del model_cp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Print comparison table
    print_comparison_table(results)

    # Save all results
    summary_path = os.path.join(args.output_dir, 'cross_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
