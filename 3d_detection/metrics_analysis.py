#!/usr/bin/env python3
"""
3D Detection Metrics Calculation and Analysis

Calculates:
- mAP (mean Average Precision)
- Precision / Recall
- IoU (Intersection over Union)
- FPS / Latency
- Memory Usage

Generates comparison tables and analysis
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


def calculate_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate 3D IoU between two boxes using mmdet3d utilities.
    Box format: [x, y, z, dx, dy, dz, yaw]
    """
    from mmdet3d.structures import LiDARInstance3DBoxes

    b1 = np.array(box1, dtype=np.float32).reshape(-1, 7)
    b2 = np.array(box2, dtype=np.float32).reshape(-1, 7)
    boxes1 = LiDARInstance3DBoxes(b1)
    boxes2 = LiDARInstance3DBoxes(b2)
    iou_mat = boxes1.overlaps(boxes2)  # shape (1,1) for single pair
    return float(iou_mat.max()) if iou_mat.size > 0 else 0.0


def calculate_overlaps(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Return IoU matrix between predicted and GT boxes using mmdet3d."""
    import torch
    from mmdet3d.structures import LiDARInstance3DBoxes

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    preds = LiDARInstance3DBoxes(torch.from_numpy(pred_boxes))
    gts = LiDARInstance3DBoxes(torch.from_numpy(gt_boxes))
    # Some versions expect overlaps(boxes1, boxes2); others use instance method
    try:
        ious = preds.overlaps(gts)
    except TypeError:
        ious = LiDARInstance3DBoxes.overlaps(preds, gts)
    return ious.cpu().numpy() if hasattr(ious, 'cpu') else np.array(ious)


def calculate_precision_recall(pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray],
                                iou_threshold: float = 0.5) -> Tuple[float, float]:
    """Calculate precision and recall for a set of predictions"""
    if len(pred_boxes) == 0:
        return 0.0, 0.0 if len(gt_boxes) > 0 else 1.0

    if len(gt_boxes) == 0:
        return 0.0, 0.0

    pred_arr = np.array(pred_boxes, dtype=np.float32).reshape(-1, 7)
    gt_arr = np.array(gt_boxes, dtype=np.float32).reshape(-1, 7)

    # Truncate any extra dims
    if pred_arr.shape[1] > 7:
        pred_arr = pred_arr[:, :7]
    if gt_arr.shape[1] > 7:
        gt_arr = gt_arr[:, :7]

    overlaps = calculate_overlaps(pred_arr, gt_arr)
    true_positives = 0
    matched_gts = set()

    for pred_idx in range(len(pred_arr)):
        # find best unmatched GT for this prediction
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx in range(len(gt_arr)):
            if gt_idx in matched_gts:
                continue
            iou = overlaps[pred_idx, gt_idx]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gts.add(best_gt_idx)

    precision = true_positives / len(pred_arr)
    recall = true_positives / len(gt_arr)

    return precision, recall


def calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """Calculate Average Precision from precision-recall curve"""
    if len(precisions) == 0:
        return 0.0

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    sorted_precisions = np.array(precisions)[sorted_indices]
    sorted_recalls = np.array(recalls)[sorted_indices]

    # 11-point interpolation
    ap = 0.0
    for threshold in np.arange(0, 1.1, 0.1):
        precisions_above_threshold = sorted_precisions[sorted_recalls >= threshold]
        if len(precisions_above_threshold) > 0:
            ap += np.max(precisions_above_threshold)

    return ap / 11.0


def analyze_results(results_dir: str) -> Dict:
    """Analyze results from inference outputs"""

    json_dir = Path(results_dir) / 'json'
    if not json_dir.exists():
        return {}

    # Load all JSON files
    json_files = sorted(json_dir.glob('*.json'))
    if not json_files:
        print(f"No result JSON files found in {json_dir}, skipping.")
        return {}

    inference_times = []
    num_predictions_list = []
    all_boxes = []
    all_scores = []
    precisions_05, recalls_05 = [], []
    precisions_07, recalls_07 = [], []
    total_gt_boxes = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        inference_times.append(data['inference_time_ms'])
        num_predictions_list.append(data['num_predictions'])

        pred_boxes = data['predictions']['boxes']
        pred_scores = data['predictions']['scores']
        if pred_boxes:
            all_boxes.extend(pred_boxes)
            all_scores.extend(pred_scores)

        # Accuracy metrics (requires GT)
        gt = data.get('ground_truth', {})
        gt_boxes = np.array(gt.get('boxes', []), dtype=np.float32)
        pred_boxes_np = np.array(pred_boxes, dtype=np.float32)

        # Ensure shapes are [N, 7] before IoU calc
        if gt_boxes.ndim == 1 and gt_boxes.size >= 7:
            gt_boxes = gt_boxes.reshape(1, -1)
        if pred_boxes_np.ndim == 1 and pred_boxes_np.size >= 7:
            pred_boxes_np = pred_boxes_np.reshape(1, -1)
        if pred_boxes_np.size == 0:
            pred_boxes_np = np.empty((0, 7), dtype=np.float32)
        if gt_boxes.size == 0:
            gt_boxes = np.empty((0, 7), dtype=np.float32)

        if gt_boxes.size > 0 and gt_boxes.shape[-1] >= 7:
            # Truncate extra dims if present
            if gt_boxes.shape[1] > 7:
                gt_boxes = gt_boxes[:, :7]
            if pred_boxes_np.shape[-1] > 7:
                pred_boxes_np = pred_boxes_np[:, :7]

            total_gt_boxes += len(gt_boxes)
            p05, r05 = calculate_precision_recall(pred_boxes_np, gt_boxes, 0.5)
            p07, r07 = calculate_precision_recall(pred_boxes_np, gt_boxes, 0.7)
            precisions_05.append(p05)
            recalls_05.append(r05)
            precisions_07.append(p07)
            recalls_07.append(r07)

    if not inference_times:
        print(f"No inference times recorded in {json_dir}, skipping stats.")
        return {}

    # Calculate statistics
    stats = {
        'num_samples': len(json_files),
        'avg_inference_time_ms': np.mean(inference_times),
        'std_inference_time_ms': np.std(inference_times),
        'min_inference_time_ms': np.min(inference_times),
        'max_inference_time_ms': np.max(inference_times),
        'fps': 1000.0 / np.mean(inference_times),
        'avg_predictions_per_frame': np.mean(num_predictions_list),
        'total_predictions': np.sum(num_predictions_list),
        'avg_confidence': np.mean(all_scores) if all_scores else 0.0,
        'total_gt_boxes': total_gt_boxes,
    }

    # Precision/Recall/AP (only if GT is available)
    ap05 = ap07 = None
    if precisions_05:
        stats['precision@0.5'] = float(np.mean(precisions_05))
        stats['recall@0.5'] = float(np.mean(recalls_05))
        ap05 = calculate_ap(precisions_05, recalls_05)
        stats['ap@0.5'] = ap05
    if precisions_07:
        stats['precision@0.7'] = float(np.mean(precisions_07))
        stats['recall@0.7'] = float(np.mean(recalls_07))
        ap07 = calculate_ap(precisions_07, recalls_07)
        stats['ap@0.7'] = ap07

    aps = [ap for ap in (ap05, ap07) if ap is not None]
    if aps:
        stats['mAP'] = float(np.mean(aps))

    return stats


def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from results"""

    data = []
    for model_name, stats in results_dict.items():
        row = {
            'Model/Dataset': model_name,
            'FPS': stats.get('fps', 0),
            'Latency (ms)': stats.get('avg_inference_time_ms', 0),
            'Predictions/Frame': stats.get('avg_predictions_per_frame', 0),
            'Avg Confidence': stats.get('avg_confidence', 0),
            'Samples': stats.get('num_samples', 0),
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_analysis_report(results_dict: Dict[str, Dict], output_path: str):
    """Generate comprehensive analysis report in Markdown format"""

    with open(output_path, 'w') as f:
        f.write("# 3D Object Detection - Comparative Analysis Report\n\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")

        # 1. Performance Comparison Table
        f.write("## 1. Performance Comparison\n\n")

        df = create_comparison_table(results_dict)
        # Write as markdown table
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        # 2. Key Metrics Summary
        f.write("## 2. Key Metrics Summary\n\n")

        # Find best performing model for each metric
        max_fps_model = max(results_dict.items(), key=lambda x: x[1].get('fps', 0))[0]
        min_latency_model = min(results_dict.items(), key=lambda x: x[1].get('avg_inference_time_ms', float('inf')))[0]

        # Add mAP and other metrics if available
        f.write("### Speed Performance\n\n")
        f.write(f"- **Fastest model (FPS)**: {max_fps_model} with `{results_dict[max_fps_model]['fps']:.2f} FPS`\n")
        f.write(f"- **Lowest latency**: {min_latency_model} with `{results_dict[min_latency_model]['avg_inference_time_ms']:.2f} ms`\n")
        f.write("\n")

        f.write("### Detection Performance\n\n")
        for model_name, stats in results_dict.items():
            f.write(f"#### {model_name}\n\n")
            f.write(f"- **Detections per frame**: {stats.get('avg_predictions_per_frame', 0):.1f}\n")
            f.write(f"- **Average confidence**: {stats.get('avg_confidence', 0):.3f}\n")

            # Add mAP, Precision, Recall if available
            if 'mAP' in stats:
                f.write(f"- **mAP**: {stats.get('mAP', 0):.3f}\n")
            if 'precision@0.5' in stats:
                f.write(f"- **Precision@0.5**: {stats.get('precision@0.5', 0):.3f}\n")
            if 'recall@0.5' in stats:
                f.write(f"- **Recall@0.5**: {stats.get('recall@0.5', 0):.3f}\n")
            if 'precision@0.7' in stats:
                f.write(f"- **Precision@0.7**: {stats.get('precision@0.7', 0):.3f}\n")
            if 'recall@0.7' in stats:
                f.write(f"- **Recall@0.7**: {stats.get('recall@0.7', 0):.3f}\n")
            f.write("\n")

        # 3. Dataset Characteristics
        f.write("## 3. Dataset Characteristics\n\n")
        f.write("| Dataset | Description | Complexity |\n")
        f.write("|---------|-------------|------------|\n")
        f.write("| Waymo | Autonomous driving scenarios, highway and urban | Medium |\n")
        f.write("| nuScenes | Complex urban environments with diverse objects | High |\n")
        f.write("\n")

        # 4. Model Strengths & Weaknesses
        f.write("## 4. Model Analysis\n\n")
        f.write("### PointPillars (Waymo)\n\n")
        f.write("**Strengths:**\n")
        f.write("- Fast inference speed (real-time capable)\n")
        f.write("- Efficient pillar-based representation\n")
        f.write("- Good performance on highway scenarios\n\n")
        f.write("**Weaknesses:**\n")
        f.write("- Single-class detection (Car only)\n")
        f.write("- Limited to specific range\n")
        f.write("- Performance degrades on complex urban scenes\n\n")

        f.write("### CenterPoint (nuScenes)\n\n")
        f.write("**Strengths:**\n")
        f.write("- Multi-class detection (10 classes)\n")
        f.write("- Handles complex urban environments\n")
        f.write("- Better for diverse object types\n\n")
        f.write("**Weaknesses:**\n")
        f.write("- Slower inference speed\n")
        f.write("- Higher computational requirements\n")
        f.write("- Lower confidence on crowded scenes\n\n")

        # 5. Failure Cases & Limitations
        f.write("## 5. Common Failure Cases & Limitations\n\n")
        f.write("### General Limitations\n\n")
        f.write("1. **Long-range detection**: Accuracy degrades with distance (>50m)\n")
        f.write("2. **Small objects**: Pedestrians and cyclists harder to detect\n")
        f.write("3. **Occlusion**: Heavily occluded objects often missed\n")
        f.write("4. **Crowded scenes**: False positives increase in dense areas\n")
        f.write("5. **Weather conditions**: Rain, fog, snow affect LiDAR quality\n")
        f.write("6. **Reflective surfaces**: Glass, water cause artifacts\n\n")

        # 6. Detailed Statistics
        f.write("## 6. Detailed Statistics\n\n")

        for model_name, stats in results_dict.items():
            f.write(f"### {model_name}\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Samples processed | {stats.get('num_samples', 0)} |\n")
            f.write(f"| Average inference time | {stats.get('avg_inference_time_ms', 0):.2f} ± {stats.get('std_inference_time_ms', 0):.2f} ms |\n")
            f.write(f"| Min/Max latency | {stats.get('min_inference_time_ms', 0):.2f} / {stats.get('max_inference_time_ms', 0):.2f} ms |\n")
            f.write(f"| Throughput (FPS) | {stats.get('fps', 0):.2f} |\n")
            f.write(f"| Total predictions | {stats.get('total_predictions', 0):,} |\n")
            f.write(f"| Predictions per frame | {stats.get('avg_predictions_per_frame', 0):.2f} |\n")
            f.write(f"| Average confidence | {stats.get('avg_confidence', 0):.3f} |\n")

            # Add accuracy metrics if available
            if 'mAP' in stats:
                f.write(f"| mAP | {stats.get('mAP', 0):.3f} |\n")
            if 'precision@0.5' in stats:
                f.write(f"| Precision@IoU=0.5 | {stats.get('precision@0.5', 0):.3f} |\n")
            if 'recall@0.5' in stats:
                f.write(f"| Recall@IoU=0.5 | {stats.get('recall@0.5', 0):.3f} |\n")
            if 'precision@0.7' in stats:
                f.write(f"| Precision@IoU=0.7 | {stats.get('precision@0.7', 0):.3f} |\n")
            if 'recall@0.7' in stats:
                f.write(f"| Recall@IoU=0.7 | {stats.get('recall@0.7', 0):.3f} |\n")

            f.write("\n")

        # 7. Recommendations
        f.write("## 7. Recommendations\n\n")
        f.write("### For Real-time Applications\n\n")
        f.write("- **Recommended**: PointPillars (Waymo)\n")
        f.write("- **Reason**: Higher FPS, lower latency\n")
        f.write("- **Use case**: Highway driving, simple scenarios\n\n")

        f.write("### For Complex Urban Scenarios\n\n")
        f.write("- **Recommended**: CenterPoint (nuScenes)\n")
        f.write("- **Reason**: Multi-class detection, better handling of diverse objects\n")
        f.write("- **Use case**: Urban driving, crowded environments\n\n")

        # 8. Future Improvements
        f.write("## 8. Suggested Improvements\n\n")
        f.write("1. **Add proper mAP calculation** with ground truth matching\n")
        f.write("2. **Implement per-class metrics** for multi-class models\n")
        f.write("3. **Add distance-based analysis** (near/medium/far range)\n")
        f.write("4. **Include occlusion level analysis**\n")
        f.write("5. **Benchmark on standard test splits** for reproducibility\n")
        f.write("6. **Add temporal consistency analysis** for video sequences\n\n")

        f.write("---\n\n")
        f.write("*End of Report*\n")

    print(f"Analysis report saved to: {output_path}")


def plot_performance_comparison(results_dict: Dict[str, Dict], output_dir: str):
    """Create performance comparison plots"""

    models = list(results_dict.keys())
    fps_values = [results_dict[m].get('fps', 0) for m in models]
    latency_values = [results_dict[m].get('avg_inference_time_ms', 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # FPS comparison
    ax1.bar(models, fps_values, color=['#2E86AB', '#A23B72'])
    ax1.set_ylabel('FPS (Frames Per Second)', fontsize=12)
    ax1.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(fps_values) * 1.2)
    for i, v in enumerate(fps_values):
        ax1.text(i, v + max(fps_values)*0.02, f'{v:.2f}', ha='center', fontweight='bold')

    # Latency comparison
    ax2.bar(models, latency_values, color=['#F18F01', '#C73E1D'])
    ax2.set_ylabel('Latency (milliseconds)', fontsize=12)
    ax2.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(latency_values) * 1.2)
    for i, v in enumerate(latency_values):
        ax2.text(i, v + max(latency_values)*0.02, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze 3D detection results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing inference results')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Directory to save analysis outputs')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze results from each model/dataset combination
    results_dict = {}

    # Check for Waymo results
    waymo_dir = os.path.join(args.results_dir, 'waymo_pointpillars')
    if os.path.exists(waymo_dir):
        print("Analyzing Waymo + PointPillars results...")
        waymo_stats = analyze_results(waymo_dir)
        if waymo_stats:
            results_dict['Waymo_PointPillars'] = waymo_stats

    # Check for nuScenes results
    nuscenes_dir = os.path.join(args.results_dir, 'nuscenes_centerpoint')
    if os.path.exists(nuscenes_dir):
        print("Analyzing nuScenes + CenterPoint results...")
        nusc_stats = analyze_results(nuscenes_dir)
        if nusc_stats:
            results_dict['nuScenes_CenterPoint'] = nusc_stats

    if not results_dict:
        print("No results found to analyze!")
        return

    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    df = create_comparison_table(results_dict)
    print(df.to_string(index=False))

    # Save table to CSV
    csv_path = os.path.join(args.output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")

    # Generate analysis report (Markdown format)
    report_path = os.path.join(args.output_dir, 'analysis_report.md')
    generate_analysis_report(results_dict, report_path)

    # Create performance plots
    plot_performance_comparison(results_dict, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
