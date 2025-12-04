#!/usr/bin/env python3
"""
Unified 3D Detection Inference, Saving, and Visualization Script

This script provides a complete solution for:
1. Running inference on multiple datasets (Waymo, nuScenes) with multiple models
2. Saving outputs: .png frames, .ply point clouds with predictions, .json metadata
3. Creating demo videos from frames
4. Open3D visualization support
5. Performance metrics calculation

Author: Assignment Solution
Date: 2025
"""

import os
import sys
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm


from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.registry import DATASETS
import mmcv
import open3d as o3d
OPEN3D_AVAILABLE = True

try:
    from mmengine.config import Config
except ImportError:
    from mmcv import Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DeepDataMiningLearning', 'detection3d'))


class PerformanceMetrics:
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.predictions = []
        self.ground_truths = []

    def add_inference_time(self, time_ms):
        self.inference_times.append(time_ms)

    def add_memory(self, memory_mb):
        self.memory_usage.append(memory_mb)

    def add_prediction(self, pred_boxes, pred_scores, pred_labels):
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })

    def add_ground_truth(self, gt_boxes, gt_labels):
        self.ground_truths.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })

    def get_fps(self):
        if not self.inference_times:
            return 0.0
        avg_time_s = np.mean(self.inference_times) / 1000.0
        return 1.0 / avg_time_s if avg_time_s > 0 else 0.0

    def get_latency_ms(self):
        if not self.inference_times:
            return 0.0, 0.0, 0.0
        return np.mean(self.inference_times), np.min(self.inference_times), np.max(self.inference_times)

    def get_memory_mb(self):
        if not self.memory_usage:
            return 0.0, 0.0
        return np.mean(self.memory_usage), np.max(self.memory_usage)


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / (1024 * 1024)


def save_point_cloud_with_boxes(points, pred_boxes, gt_boxes, save_path, pred_labels=None, pred_scores=None):
    """
    Save point cloud with prediction and GT boxes to PLY file

    Args:
        points: (N, 3+) point cloud
        pred_boxes: (M, 7) predicted boxes [x, y, z, dx, dy, dz, yaw]
        gt_boxes: (K, 7) ground truth boxes
        save_path: output .ply file path
        pred_labels: (M,) predicted class labels
        pred_scores: (M,) prediction confidence scores
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if points.shape[1] >= 4:
        z_values = points[:, 2]
        z_norm = (z_values - z_values.min()) / (z_values.ptp() + 1e-6)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_norm
        colors[:, 2] = 1 - z_norm
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # Add predicted boxes (green)
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            line_set = create_bbox_line_set(box, color=[0, 1, 0])  # Green
            geometries.append(line_set)

    # Add GT boxes (red)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            line_set = create_bbox_line_set(box, color=[1, 0, 0])  # Red
            geometries.append(line_set)

    # Save combined geometry
    all_points = np.asarray(pcd.points)
    all_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Merge all line sets
    for geom in geometries[1:]:
        if isinstance(geom, o3d.geometry.LineSet):
            geom_points = np.asarray(geom.points)
            all_points = np.vstack([all_points, geom_points])
            if all_colors is not None:
                geom_colors = np.asarray(geom.colors)
                # Line colors need to be expanded to point colors
                if len(geom_colors) > 0:
                    point_colors = np.tile(geom_colors[0], (len(geom_points), 1))
                    all_colors = np.vstack([all_colors, point_colors])

    # Save merged point cloud
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(all_points)
    if all_colors is not None:
        merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud(save_path, merged_pcd)
    print(f"Saved PLY: {save_path}")


def create_bbox_line_set(box, color=[0, 1, 0]):
    x, y, z, dx, dy, dz, yaw = box[:7]

    corners = np.array([
        [dx/2, dy/2, dz/2],
        [dx/2, -dy/2, dz/2],
        [-dx/2, -dy/2, dz/2],
        [-dx/2, dy/2, dz/2],
        [dx/2, dy/2, -dz/2],
        [dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2, dy/2, -dz/2],
    ])

    # Rotate
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners @ R.T

    # Translate
    corners += np.array([x, y, z])

    # Create line set
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors_array = np.tile(color, (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors_array)

    return line_set


def draw_boxes_on_image(image, boxes_2d, labels, scores=None, color=(0, 255, 0)):
    img = image.copy()

    for i, box in enumerate(boxes_2d):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = labels[i] if i < len(labels) else "Object"
        if scores is not None and i < len(scores):
            label = f"{label}: {scores[i]:.2f}"

        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 2)

    return img


def create_video_from_frames(frame_dir, output_path, fps=10):
    """Create video from saved frame images"""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    h, w = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_file in tqdm(frame_files, desc="Creating video"):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Video saved: {output_path}")


def run_inference_on_dataset(model, config_path, dataset_name,
                            data_root, output_dir,
                            num_samples=10, save_ply=True,
                            save_vis=True):
    """
    Run inference on a dataset and save outputs

    Args:
        model: MMDet3D model
        config_path: path to config file
        dataset_name: 'waymo' or 'nuscenes'
        data_root: dataset root directory
        output_dir: where to save results
        num_samples: number of samples to process
        save_ply: whether to save .ply files
        save_vis: whether to save visualization images

    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    ply_dir = os.path.join(output_dir, 'ply')
    vis_dir = os.path.join(output_dir, 'frames')
    json_dir = os.path.join(output_dir, 'json')

    if save_ply:
        os.makedirs(ply_dir, exist_ok=True)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # Load config
    cfg = Config.fromfile(config_path)

    # Get test dataset
    if dataset_name == 'waymo':
        test_data_path = os.path.join(data_root, 'kitti_format/testing/velodyne')
        all_files = sorted(list(Path(test_data_path).glob('*.bin')))
        sample_files = all_files if num_samples <= 0 else all_files[:num_samples]
    elif dataset_name == 'nuscenes':
        # Load nuScenes test data
        test_data_path = os.path.join(data_root, 'samples/LIDAR_TOP')
        all_files = sorted(list(Path(test_data_path).glob('*.pcd.bin')))
        sample_files = all_files if num_samples <= 0 else all_files[:num_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nProcessing {len(sample_files)} samples from {dataset_name}...")

    for idx, sample_file in enumerate(tqdm(sample_files, desc=f"Inference on {dataset_name}")):
        try:
            # Load point cloud
            if dataset_name == 'waymo':
                # Waymo KITTI format: x, y, z, intensity, elongation, timestamp (6D)
                # We use first 4 dimensions for compatibility
                points_raw = np.fromfile(sample_file, dtype=np.float32)
                # Auto-detect dimension (usually 6 for Waymo)
                if len(points_raw) % 6 == 0:
                    points = points_raw.reshape(-1, 6)[:, :4]  # Take x,y,z,intensity
                elif len(points_raw) % 5 == 0:
                    points = points_raw.reshape(-1, 5)[:, :4]
                elif len(points_raw) % 4 == 0:
                    points = points_raw.reshape(-1, 4)
                else:
                    raise ValueError(f"Unexpected point cloud size: {len(points_raw)}")
            else:  # nuscenes
                points = np.fromfile(sample_file, dtype=np.float32).reshape(-1, 5)

            # Prepare input
            data_dict = {
                'points': points,
                'timestamp': time.time(),
                'sample_idx': idx
            }

            # Run inference with timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            result = inference_detector(model, str(sample_file))

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000
            metrics.add_inference_time(inference_time_ms)
            metrics.add_memory(get_gpu_memory_mb())

            # Handle different return types (tuple or single object)
            if isinstance(result, (tuple, list)):
                result = result[0]

            pred_instances = result.pred_instances_3d
            pred_boxes = pred_instances.bboxes_3d.tensor.cpu().numpy()
            pred_scores = pred_instances.scores_3d.cpu().numpy()
            pred_labels = pred_instances.labels_3d.cpu().numpy()

            metrics.add_prediction(pred_boxes, pred_scores, pred_labels)

            # Save JSON metadata
            metadata = {
                'sample_idx': idx,
                'sample_file': str(sample_file),
                'dataset': dataset_name,
                'inference_time_ms': float(inference_time_ms),
                'num_predictions': int(len(pred_boxes)),
                'predictions': {
                    'boxes': pred_boxes.tolist(),
                    'scores': pred_scores.tolist(),
                    'labels': pred_labels.tolist()
                }
            }

            json_path = os.path.join(json_dir, f'sample_{idx:06d}.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save PLY with boxes
            if save_ply and OPEN3D_AVAILABLE:
                ply_path = os.path.join(ply_dir, f'sample_{idx:06d}.ply')
                save_point_cloud_with_boxes(points, pred_boxes, np.array([]), ply_path,
                                           pred_labels, pred_scores)

            # Save visualization (BEV view)
            if save_vis:
                vis_img = create_bev_visualization(points, pred_boxes, pred_labels, pred_scores)
                vis_path = os.path.join(vis_dir, f'frame_{idx:06d}.png')
                cv2.imwrite(vis_path, vis_img)

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            traceback.print_exc()
            continue

    return metrics


def create_bev_visualization(points, boxes, labels, scores,
                            img_size=800, scale=5.0):
    """Create bird's eye view visualization"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    center = img_size // 2

    # Draw points
    if len(points) > 0:
        pts_2d = points[:, :2] * scale + center
        pts_2d = pts_2d.astype(int)
        mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_size) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_size)
        pts_2d = pts_2d[mask]

        for pt in pts_2d:
            cv2.circle(img, tuple(pt), 1, (100, 100, 100), -1)

    # Draw boxes
    for i, box in enumerate(boxes):
        if len(box) >= 7:
            x, y, z, dx, dy, dz, yaw = box[:7]
        else:
            continue

        corners = np.array([
            [dx/2, dy/2],
            [dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2, dy/2],
        ])

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T
        corners += np.array([x, y])

        corners = corners * scale + center
        corners = corners.astype(int)

        color = (0, 255, 0) if i < len(scores) and scores[i] > 0.5 else (0, 200, 200)
        cv2.polylines(img, [corners], True, color, 2)

        # Draw direction indicator
        front_center = ((corners[0] + corners[1]) // 2).astype(int)
        box_center = corners.mean(axis=0).astype(int)
        cv2.arrowedLine(img, tuple(box_center), tuple(front_center), color, 2)

    return img


def calculate_metrics_summary(metrics_dict):
    summary = {}

    for name, metrics in metrics_dict.items():
        fps = metrics.get_fps()
        latency_mean, latency_min, latency_max = metrics.get_latency_ms()
        mem_mean, mem_max = metrics.get_memory_mb()

        summary[name] = {
            'FPS': fps,
            'Latency (ms)': {
                'mean': latency_mean,
                'min': latency_min,
                'max': latency_max
            },
            'Memory (MB)': {
                'mean': mem_mean,
                'max': mem_max
            },
            'Total Samples': len(metrics.predictions)
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified 3D Detection Inference and Visualization')

    # Model and dataset arguments
    parser.add_argument('--waymo-config', type=str,
                       default='checkpoints/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-car.py',
                       help='Config file for Waymo model')
    parser.add_argument('--waymo-checkpoint', type=str,
                       default='checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth',
                       help='Checkpoint file for Waymo model')
    parser.add_argument('--waymo-data', type=str,
                       default='data_waymo_kitti',
                       help='Waymo dataset root')

    parser.add_argument('--nuscenes-config', type=str,
                       default='checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
                       help='Config file for nuScenes model')
    parser.add_argument('--nuscenes-checkpoint', type=str,
                       default='checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth',
                       help='Checkpoint file for nuScenes model')
    parser.add_argument('--nuscenes-data', type=str,
                       default='data_nuscenes',
                       help='nuScenes dataset root')

    # Processing arguments
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to process per dataset (use 0 or -1 for all samples)')
    parser.add_argument('--waymo-samples', type=int, default=None,
                       help='Number of Waymo samples (overrides --num-samples for Waymo)')
    parser.add_argument('--nuscenes-samples', type=int, default=None,
                       help='Number of nuScenes samples (overrides --num-samples for nuScenes)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    parser.add_argument('--save-ply', action='store_true', default=True,
                       help='Save point clouds as PLY files')
    parser.add_argument('--save-vis', action='store_true', default=True,
                       help='Save visualization frames')
    parser.add_argument('--create-video', action='store_true', default=True,
                       help='Create demo video from frames')
    parser.add_argument('--fps', type=int, default=10,
                       help='FPS for output video')

    args = parser.parse_args()

    # Store all metrics
    all_metrics = {}

    # Process Waymo dataset
    print("\n" + "="*80)
    print("Processing Waymo Dataset with PointPillars")
    print("="*80)

    try:
        waymo_model = init_model(args.waymo_config, args.waymo_checkpoint, device=args.device)
        waymo_output = os.path.join(args.output_dir, 'waymo_pointpillars')
        waymo_num_samples = args.waymo_samples if args.waymo_samples is not None else args.num_samples
        waymo_metrics = run_inference_on_dataset(
            waymo_model, args.waymo_config, 'waymo', args.waymo_data,
            waymo_output, waymo_num_samples, args.save_ply, args.save_vis
        )
        all_metrics['Waymo_PointPillars'] = waymo_metrics

        if args.create_video:
            video_path = os.path.join(waymo_output, 'demo_waymo.mp4')
            create_video_from_frames(os.path.join(waymo_output, 'frames'), video_path, args.fps)

    except Exception as e:
        print(f"Error processing Waymo: {e}")
        traceback.print_exc()

    # Process nuScenes dataset
    print("\n" + "="*80)
    print("Processing nuScenes Dataset with CenterPoint")
    print("="*80)

    try:
        nuscenes_model = init_model(args.nuscenes_config, args.nuscenes_checkpoint, device=args.device)
        nuscenes_output = os.path.join(args.output_dir, 'nuscenes_centerpoint')
        nuscenes_num_samples = args.nuscenes_samples if args.nuscenes_samples is not None else args.num_samples
        nuscenes_metrics = run_inference_on_dataset(
            nuscenes_model, args.nuscenes_config, 'nuscenes', args.nuscenes_data,
            nuscenes_output, nuscenes_num_samples, args.save_ply, args.save_vis
        )
        all_metrics['nuScenes_CenterPoint'] = nuscenes_metrics

        if args.create_video:
            video_path = os.path.join(nuscenes_output, 'demo_nuscenes.mp4')
            create_video_from_frames(os.path.join(nuscenes_output, 'frames'), video_path, args.fps)

    except Exception as e:
        print(f"Error processing nuScenes: {e}")
        traceback.print_exc()

    # Generate summary report
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    summary = calculate_metrics_summary(all_metrics)

    # Print summary table
    print(f"\n{'Model/Dataset':<30} {'FPS':<10} {'Latency(ms)':<15} {'Memory(MB)':<15}")
    print("-" * 70)

    for name, stats in summary.items():
        print(f"{name:<30} {stats['FPS']:<10.2f} {stats['Latency (ms)']['mean']:<15.2f} "
              f"{stats['Memory (MB)']['mean']:<15.2f}")

    # Save summary to JSON
    summary_path = os.path.join(args.output_dir, 'performance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
