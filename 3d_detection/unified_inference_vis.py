import os
import sys
import argparse
import json
import time
import traceback
from pathlib import Path
import pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm


from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.registry import DATASETS
import mmcv
import open3d as o3d

try:
    from mmengine.config import Config
except ImportError:
    from mmcv import Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DeepDataMiningLearning', 'detection3d'))

CAM_TO_FOLDER = {
    'CAM_FRONT': 'image_0',
    'CAM_FRONT_LEFT': 'image_1',
    'CAM_FRONT_RIGHT': 'image_2',
    'CAM_SIDE_LEFT': 'image_3',
    'CAM_SIDE_RIGHT': 'image_4',
}

NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
NUSCENES_CLASS_TO_ID = {name: idx for idx, name in enumerate(NUSCENES_CLASSES)}


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


def get_pred_color(score=None):
    # Bright yellow for confident boxes, orange for low scores
    return (0, 255, 255) if score is None or score >= 0.5 else (0, 165, 255)


def draw_3d_boxes_on_camera(image, boxes, scores, lidar2img):
    """Draw 3D bounding boxes projected on camera image using calibration"""
    img = image.copy()
    h, w = img.shape[:2]

    lidar2img = np.asarray(lidar2img)
    if lidar2img.shape != (3, 4) and lidar2img.shape != (4, 4):
        print(f"Warning: Invalid lidar2img shape: {lidar2img.shape}")
        return img

    # Convert 4x4 to 3x4 if needed
    if lidar2img.shape == (4, 4):
        lidar2img = lidar2img[:3, :]

    num_boxes_drawn = 0
    num_boxes_filtered = 0

    for i, box in enumerate(boxes):
        if len(box) < 7:
            continue

        # Get 3D corners
        corners_3d = get_3d_box_corners(box)
        corners_h = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
        projected = corners_h @ lidar2img.T
        depth = projected[:, 2:3]

        # Strict depth check - all corners must be in front of camera
        valid_depth = depth > 0.1
        if not np.all(valid_depth):
            num_boxes_filtered += 1
            continue

        corners_2d = projected[:, :2] / (depth + 1e-6)

        # Proper visibility check - require reasonable portion of box to be in view
        x_coords = corners_2d[:, 0]
        y_coords = corners_2d[:, 1]

        # Check if box center is reasonably within image bounds (with small margin)
        center_x = x_coords.mean()
        center_y = y_coords.mean()
        margin_x = w * 0.3  # 30% margin
        margin_y = h * 0.3

        if not (-margin_x < center_x < w + margin_x and -margin_y < center_y < h + margin_y):
            num_boxes_filtered += 1
            continue

        # Also check if at least 2 corners are within reasonable bounds
        x_valid = (x_coords > -w * 0.5) & (x_coords < w * 1.5)
        y_valid = (y_coords > -h * 0.5) & (y_coords < h * 1.5)
        valid_corners = x_valid & y_valid

        if np.sum(valid_corners) < 2:
            num_boxes_filtered += 1
            continue

        # Draw 3D box
        score = scores[i] if scores is not None and i < len(scores) else None
        color = get_pred_color(score)

        corners_2d_int = corners_2d.astype(int)

        # Draw bottom face
        for j in range(4):
            pt1 = tuple(corners_2d_int[j])
            pt2 = tuple(corners_2d_int[(j+1)%4])
            cv2.line(img, pt1, pt2, color, 2)

        # Draw top face
        for j in range(4, 8):
            pt1 = tuple(corners_2d_int[j])
            pt2 = tuple(corners_2d_int[4 + (j+1)%4])
            cv2.line(img, pt1, pt2, color, 2)

        # Draw vertical lines
        for j in range(4):
            pt1 = tuple(corners_2d_int[j])
            pt2 = tuple(corners_2d_int[j+4])
            cv2.line(img, pt1, pt2, color, 2)

        # Draw label
        if scores is not None and i < len(scores):
            label_text = f"{scores[i]:.2f}"
            center_2d = corners_2d.mean(axis=0).astype(int)
            # Clamp label position to image bounds
            label_x = max(10, min(w-100, center_2d[0]))
            label_y = max(20, min(h-20, center_2d[1]))
            cv2.putText(img, label_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        num_boxes_drawn += 1

    if num_boxes_drawn == 0 and len(boxes) > 0:
        print(f"Warning: Drew {num_boxes_drawn}/{len(boxes)} boxes on camera (filtered: {num_boxes_filtered})")

    # Add legend with background for better visibility
    legend_y = 25
    cv2.rectangle(img, (5, 5), (250, 75), (0, 0, 0), -1)  # Black background
    cv2.rectangle(img, (5, 5), (250, 75), (255, 255, 255), 2)  # White border
    cv2.putText(img, "3D Detection (Camera)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Prediction (conf>=0.5)", (10, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_pred_color(1.0), 2)
    cv2.putText(img, "Prediction (conf<0.5)", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_pred_color(0.0), 2)

    # Add detection count
    if num_boxes_drawn > 0:
        count_text = f"Detections: {num_boxes_drawn}"
        cv2.rectangle(img, (w-180, 5), (w-5, 35), (0, 0, 0), -1)
        cv2.rectangle(img, (w-180, 5), (w-5, 35), (255, 255, 255), 2)
        cv2.putText(img, count_text, (w-170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


def get_3d_box_corners(box):
    """Get 8 corners of 3D bounding box"""
    x, y, z, dx, dy, dz, yaw = box[:7]

    # Create box corners in object coordinate system
    corners = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ])

    # Rotation matrix around z-axis
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # Rotate and translate
    corners = corners @ R.T
    corners += np.array([x, y, z])

    return corners


def create_video_from_frames(frame_dir, output_path, fps=10, pattern='*.png'):
    """Create video from saved frame images"""
    import glob
    if '*' in pattern:
        frame_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(frame_dir, pattern))])
    else:
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    if not frame_files:
        print(f"No frames found in {frame_dir} with pattern {pattern}")
        return

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


def create_side_by_side_video(frame_dir, output_path, fps=10,
                              bev_pattern='frame_*.png',
                              cam_pattern='camera_*.png'):
    """Combine camera and BEV frames into a single video."""
    import glob
    bev_files = sorted(glob.glob(os.path.join(frame_dir, bev_pattern)))
    cam_files = sorted(glob.glob(os.path.join(frame_dir, cam_pattern)))

    if not bev_files or not cam_files:
        print(f"Skipping combined video: bev ({len(bev_files)}) or camera ({len(cam_files)}) frames missing")
        return

    num_frames = min(len(bev_files), len(cam_files))
    bev_files = bev_files[:num_frames]
    cam_files = cam_files[:num_frames]

    first_cam = cv2.imread(cam_files[0])
    first_bev = cv2.imread(bev_files[0])
    if first_cam is None or first_bev is None:
        print("Skipping combined video: could not read first frames")
        return

    cam_h, cam_w = first_cam.shape[:2]
    bev_h, bev_w = first_bev.shape[:2]
    bev_w_resized = int(bev_w * (cam_h / bev_h))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (cam_w + bev_w_resized, cam_h))

    for bev_file, cam_file in tqdm(zip(bev_files, cam_files), total=num_frames, desc="Creating combined video"):
        cam_img = cv2.imread(cam_file)
        bev_img = cv2.imread(bev_file)
        if cam_img is None or bev_img is None:
            continue
        bev_img = cv2.resize(bev_img, (bev_w_resized, cam_h))
        combined = np.hstack([cam_img, bev_img])
        out.write(combined)

    out.release()
    print(f"Combined video saved: {output_path}")


def load_waymo_info_map(data_root, target_lidar_files):
    """Load Waymo info entries for the given lidar file names."""
    target_lidar_files = set(target_lidar_files)
    info_files = [
        'waymo_infos_train.pkl',
        'waymo_infos_trainval.pkl',
        'waymo_infos_val.pkl',
        'waymo_infos_test.pkl',
    ]
    info_map = {}
    base = Path(data_root) / 'kitti_format'

    for name in info_files:
        path = base / name
        if not path.exists():
            continue
        try:
            data = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue
        data_list = data['data_list'] if isinstance(data, dict) and 'data_list' in data else data
        for entry in data_list:
            lid = entry['lidar_points']['lidar_path']
            if lid in target_lidar_files:
                info_map[lid] = entry
                if len(info_map) == len(target_lidar_files):
                    return info_map

    missing = target_lidar_files - set(info_map.keys())
    if missing:
        print(f"Calibration missing for {len(missing)} samples (no matching info entry).")
    return info_map


def load_nuscenes_info_map(data_root, target_lidar_files):
    """Load nuScenes info entries for the given lidar file names."""
    target_lidar_files = set(target_lidar_files)
    info_files = [
        'nuscenes_infos_val.pkl',
        'nuscenes_infos_train.pkl',
        'nuscenes_infos_trainval.pkl',
    ]
    info_map = {}
    base = Path(data_root)

    for name in info_files:
        path = base / name
        if not path.exists():
            continue
        try:
            data = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue
        if isinstance(data, dict):
            data_list = data.get('data_list') or data.get('infos', [])
        else:
            data_list = data
        for entry in data_list:
            lidar_path = entry.get('lidar_path') or entry.get('lidar_points', {}).get('lidar_path', '')
            lid = Path(lidar_path).name
            if lid in target_lidar_files:
                info_map[lid] = entry
                if len(info_map) == len(target_lidar_files):
                    return info_map

    missing = target_lidar_files - set(info_map.keys())
    if missing:
        print(f"nuScenes info missing for {len(missing)} samples (no matching info entry).")
    return info_map


def run_inference_on_dataset(model, config_path, dataset_name,
                            data_root, output_dir,
                            num_samples=10, save_ply=True,
                            save_vis=True, require_gt=False):
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
    lidar_info_map = {}
    nuscenes_info_map = {}

    if dataset_name == 'waymo':
        # For mAP we need GT, so prefer training split when require_gt is True.
        test_data_path = os.path.join(data_root, 'kitti_format/testing/velodyne')
        train_data_path = os.path.join(data_root, 'kitti_format/training/velodyne')
        data_path = train_data_path if require_gt and os.path.exists(train_data_path) else test_data_path
        if not os.path.exists(data_path) and os.path.exists(test_data_path):
            data_path = test_data_path
        all_files = sorted(list(Path(data_path).glob('*.bin')))
        sample_files = all_files if num_samples <= 0 else all_files[:num_samples]
        # Load Waymo info map once to reuse for calibration and GT lookup
        lidar_info_map = load_waymo_info_map(
            data_root, [Path(f).name for f in sample_files])
    elif dataset_name == 'nuscenes':
        # Load nuScenes test data
        test_data_path = os.path.join(data_root, 'samples/LIDAR_TOP')
        all_files = sorted(list(Path(test_data_path).glob('*.pcd.bin')))
        sample_files = all_files if num_samples <= 0 else all_files[:num_samples]
        nuscenes_info_map = load_nuscenes_info_map(
            data_root, [Path(f).name for f in sample_files])
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
            if pred_boxes.ndim == 2 and pred_boxes.shape[1] > 7:
                pred_boxes = pred_boxes[:, :7]
            pred_scores = pred_instances.scores_3d.cpu().numpy()
            pred_labels = pred_instances.labels_3d.cpu().numpy()

            metrics.add_prediction(pred_boxes, pred_scores, pred_labels)

            # Lookup GT boxes if available (Waymo info PKL)
            gt_boxes = np.empty((0, 7))
            gt_labels = np.empty((0,), dtype=np.int64)
            if dataset_name == 'waymo':
                lid_name = Path(sample_file).name
                info_entry = lidar_info_map.get(lid_name, {})
                instances = info_entry.get('instances', []) if isinstance(info_entry, dict) else []
                if instances:
                    gt_boxes_list = []
                    gt_labels_list = []
                    for inst in instances:
                        box = inst.get('bbox_3d', None)
                        label = inst.get('bbox_label_3d', None)
                        if box is None:
                            continue
                        gt_boxes_list.append(box)
                        gt_labels_list.append(label if label is not None else -1)
                    if gt_boxes_list:
                        gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
                        if gt_boxes.shape[1] > 7:
                            gt_boxes = gt_boxes[:, :7]
                        gt_labels = np.array(gt_labels_list, dtype=np.int64)
            elif dataset_name == 'nuscenes':
                lid_name = Path(sample_file).name
                info_entry = nuscenes_info_map.get(lid_name, {})
                if info_entry:
                    boxes = info_entry.get('gt_boxes', [])
                    names = info_entry.get('gt_names', [])
                    if boxes is not None and len(boxes) > 0:
                        gt_boxes = np.array(boxes, dtype=np.float32)
                        if gt_boxes.ndim == 2 and gt_boxes.shape[1] > 7:
                            gt_boxes = gt_boxes[:, :7]
                        # Map class names to ids for consistency; unknown -> -1
                        gt_labels = np.array([NUSCENES_CLASS_TO_ID.get(n, -1) for n in names], dtype=np.int64)
            metrics.add_ground_truth(gt_boxes, gt_labels)

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
                },
                'ground_truth': {
                    'boxes': gt_boxes.tolist(),
                    'labels': gt_labels.tolist()
                }
            }

            json_path = os.path.join(json_dir, f'sample_{idx:06d}.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save PLY with boxes

            ply_path = os.path.join(ply_dir, f'sample_{idx:06d}.ply')
            save_point_cloud_with_boxes(points, pred_boxes, gt_boxes, ply_path,
                                        pred_labels, pred_scores)

            # Save visualization (BEV view)
            if save_vis:
                vis_img = create_bev_visualization(points, pred_boxes, pred_labels, pred_scores)
                vis_path = os.path.join(vis_dir, f'frame_{idx:06d}.png')
                cv2.imwrite(vis_path, vis_img)

                # Save camera view if available
                if dataset_name == 'waymo':
                    sample_id = Path(sample_file).stem
                    lidar_key = f"{sample_id}.bin"
                    info_entry = lidar_info_map.get(lidar_key, None)
                    cam_name = 'CAM_FRONT'

                    if info_entry and 'images' in info_entry and cam_name in info_entry['images']:
                        cam_info = info_entry['images'][cam_name]
                        img_name = Path(cam_info['img_path']).name
                        split_dir = 'testing' if sample_id.startswith('2') else 'training'
                        cam_folder = CAM_TO_FOLDER.get(cam_name, 'image_0')
                        img_path = Path(data_root) / 'kitti_format' / split_dir / cam_folder / img_name

                        if img_path.exists():
                            camera_img = cv2.imread(str(img_path))
                            if camera_img is not None:
                                lidar2img = cam_info.get('lidar2img')
                                camera_vis = draw_3d_boxes_on_camera(camera_img, pred_boxes, pred_scores, lidar2img)
                                camera_path = os.path.join(vis_dir, f'camera_{idx:06d}.png')
                                cv2.imwrite(camera_path, camera_vis)

                elif dataset_name == 'nuscenes':
                    lid_name = Path(sample_file).name
                    info_entry = nuscenes_info_map.get(lid_name, {})
                    cam_name = 'CAM_FRONT'
                    camera_saved = False

                    if info_entry and 'cams' in info_entry and cam_name in info_entry['cams']:
                        cam_info = info_entry['cams'][cam_name]
                        img_path = Path(data_root) / cam_info.get('data_path', '').replace('./data/nuscenes/', '')
                        if img_path.exists():
                            camera_img = cv2.imread(str(img_path))
                            if camera_img is not None:
                                cam_intrinsic = np.array(cam_info.get('cam_intrinsic', np.eye(3)))
                                s2l_R = np.array(cam_info.get('sensor2lidar_rotation', np.eye(3)))
                                s2l_t = np.array(cam_info.get('sensor2lidar_translation', np.zeros(3)))
                                # lidar to camera: invert sensor2lidar
                                lidar2cam_R = s2l_R.T
                                lidar2cam_t = -lidar2cam_R @ s2l_t
                                lidar2cam = np.eye(4)
                                lidar2cam[:3, :3] = lidar2cam_R
                                lidar2cam[:3, 3] = lidar2cam_t
                                lidar2img = cam_intrinsic @ lidar2cam[:3, :]
                                camera_vis = draw_3d_boxes_on_camera(camera_img, pred_boxes, pred_scores, lidar2img)
                                camera_path = os.path.join(vis_dir, f'camera_{idx:06d}.png')
                                cv2.imwrite(camera_path, camera_vis)
                                camera_saved = True

                    # Fallback: simple image overlay without projection if calibration missing
                    if not camera_saved:
                        img_path = Path(data_root) / 'images' / f'sample_{idx:06d}.jpg'
                        if img_path.exists():
                            camera_img = cv2.imread(str(img_path))
                            if camera_img is not None:
                                h, w = camera_img.shape[:2]
                                cv2.rectangle(camera_img, (5, 5), (400, 50), (0, 0, 0), -1)
                                cv2.rectangle(camera_img, (5, 5), (400, 50), (255, 255, 255), 2)
                                cv2.putText(camera_img, "Camera View (No calibration)", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                camera_path = os.path.join(vis_dir, f'camera_{idx:06d}.png')
                                cv2.imwrite(camera_path, camera_img)
                                camera_saved = True

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

        score = scores[i] if i < len(scores) else None
        color = get_pred_color(score)
        cv2.polylines(img, [corners], True, color, 2)

        # Draw direction indicator
        front_center = ((corners[0] + corners[1]) // 2).astype(int)
        box_center = corners.mean(axis=0).astype(int)
        cv2.arrowedLine(img, tuple(box_center), tuple(front_center), color, 2)

    # Add legend with background for better visibility
    cv2.rectangle(img, (5, 5), (250, 75), (0, 0, 0), -1)  # Black background
    cv2.rectangle(img, (5, 5), (250, 75), (255, 255, 255), 2)  # White border
    cv2.putText(img, "BEV Detection (Top)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Prediction (conf>=0.5)", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_pred_color(1.0), 2)
    cv2.putText(img, "Prediction (conf<0.5)", (10, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_pred_color(0.0), 2)

    # Add detection count
    if len(boxes) > 0:
        count_text = f"Detections: {len(boxes)}"
        cv2.rectangle(img, (img_size-180, 5), (img_size-5, 35), (0, 0, 0), -1)
        cv2.rectangle(img, (img_size-180, 5), (img_size-5, 35), (255, 255, 255), 2)
        cv2.putText(img, count_text, (img_size-170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
    parser.add_argument('--require-gt', action='store_true', default=False,
                       help='If set, will try to load ground truth (needed for mAP)')
    parser.add_argument('--skip-waymo', action='store_true', default=False,
                       help='Skip Waymo processing')
    parser.add_argument('--skip-nuscenes', action='store_true', default=False,
                       help='Skip nuScenes processing')

    args = parser.parse_args()

    # Store all metrics
    all_metrics = {}

    # Process Waymo dataset
    if not args.skip_waymo:
        print("\n" + "="*80)
        print("Processing Waymo Dataset with PointPillars")
        print("="*80)

        try:
            waymo_model = init_model(args.waymo_config, args.waymo_checkpoint, device=args.device)
            waymo_output = os.path.join(args.output_dir, 'waymo_pointpillars')
            waymo_num_samples = args.waymo_samples if args.waymo_samples is not None else args.num_samples
            waymo_metrics = run_inference_on_dataset(
                waymo_model, args.waymo_config, 'waymo', args.waymo_data,
                waymo_output, waymo_num_samples, args.save_ply, args.save_vis,
                require_gt=args.require_gt
            )
            all_metrics['Waymo_PointPillars'] = waymo_metrics

            if args.create_video:
                frames_dir = os.path.join(waymo_output, 'frames')

                # Check if camera frames are available
                camera_frames = [f for f in os.listdir(frames_dir) if f.startswith('camera_')]

                if camera_frames:
                    # Create combined video as the main output
                    combined_video_path = os.path.join(waymo_output, 'demo_waymo.mp4')
                    create_side_by_side_video(frames_dir, combined_video_path, args.fps)
                    print(f"\n*** Main output: {combined_video_path} ***")

                    # Also create individual videos for reference
                    bev_video_path = os.path.join(waymo_output, 'demo_waymo_bev.mp4')
                    create_video_from_frames(frames_dir, bev_video_path, args.fps, pattern='frame_*.png')

                    camera_video_path = os.path.join(waymo_output, 'demo_waymo_camera.mp4')
                    create_video_from_frames(frames_dir, camera_video_path, args.fps, pattern='camera_*.png')
                else:
                    # Only BEV available
                    video_path = os.path.join(waymo_output, 'demo_waymo_bev.mp4')
                    create_video_from_frames(frames_dir, video_path, args.fps, pattern='frame_*.png')
                    print(f"\n*** Main output: {video_path} (camera frames not available) ***")

        except Exception as e:
            print(f"Error processing Waymo: {e}")
            traceback.print_exc()

    # Process nuScenes dataset
    if not args.skip_nuscenes:
        print("\n" + "="*80)
        print("Processing nuScenes Dataset with CenterPoint")
        print("="*80)

        try:
            nuscenes_model = init_model(args.nuscenes_config, args.nuscenes_checkpoint, device=args.device)
            nuscenes_output = os.path.join(args.output_dir, 'nuscenes_centerpoint')
            nuscenes_num_samples = args.nuscenes_samples if args.nuscenes_samples is not None else args.num_samples
            nuscenes_metrics = run_inference_on_dataset(
                nuscenes_model, args.nuscenes_config, 'nuscenes', args.nuscenes_data,
                nuscenes_output, nuscenes_num_samples, args.save_ply, args.save_vis,
                require_gt=args.require_gt
            )
            all_metrics['nuScenes_CenterPoint'] = nuscenes_metrics

            if args.create_video:
                frames_dir = os.path.join(nuscenes_output, 'frames')

                # Check if camera frames are available
                camera_frames = [f for f in os.listdir(frames_dir) if f.startswith('camera_')]

                if camera_frames:
                    # Create combined video as the main output
                    combined_video_path = os.path.join(nuscenes_output, 'demo_nuscenes.mp4')
                    create_side_by_side_video(frames_dir, combined_video_path, args.fps)
                    print(f"\n*** Main output: {combined_video_path} ***")

                    # Also create individual videos for reference
                    bev_video_path = os.path.join(nuscenes_output, 'demo_nuscenes_bev.mp4')
                    create_video_from_frames(frames_dir, bev_video_path, args.fps, pattern='frame_*.png')

                    camera_video_path = os.path.join(nuscenes_output, 'demo_nuscenes_camera.mp4')
                    create_video_from_frames(frames_dir, camera_video_path, args.fps, pattern='camera_*.png')
                else:
                    # Only BEV available
                    video_path = os.path.join(nuscenes_output, 'demo_nuscenes_bev.mp4')
                    create_video_from_frames(frames_dir, video_path, args.fps, pattern='frame_*.png')
                    print(f"\n*** Main output: {video_path} (camera frames not available) ***")

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


# TODO: have to add camera video detection model (good with color maybe?)
if __name__ == '__main__':
    main()
