"""
CustomRCNN 출력 해석 및 시각화 예제
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Waymo 데이터셋 클래스 매핑
WAYMO_CLASSES = {
    0: 'background',  # postprocess에서 제거되므로 실제론 안 나옴
    1: 'Vehicles',
    2: 'Pedestrians',
    3: 'Cyclists',
    4: 'Signs'
}

# COCO 데이터셋 클래스 매핑 (90 classes)
COCO_CLASSES = {
    0: 'background',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # ... (생략)
}


def interpret_detection(detection, class_names, score_threshold=0.5):
    """
    Detection 결과를 해석합니다.

    Args:
        detection: Dict with 'boxes', 'labels', 'scores'
        class_names: Dict mapping label ID to class name
        score_threshold: 최소 confidence 점수 (기본 0.5)

    Returns:
        filtered_detections: List[Dict] - 필터링된 detection 리스트
    """
    boxes = detection['boxes'].cpu().numpy()
    labels = detection['labels'].cpu().numpy()
    scores = detection['scores'].cpu().numpy()

    print(f"\n{'='*60}")
    print(f"전체 Detection 개수: {len(boxes)}")
    print(f"Score 범위: {scores.min():.3f} ~ {scores.max():.3f}")
    print(f"{'='*60}\n")

    # Score threshold로 필터링
    mask = scores >= score_threshold
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]

    print(f"Threshold {score_threshold} 이상: {len(filtered_boxes)}개\n")

    # 결과 해석
    filtered_detections = []
    for i, (box, label, score) in enumerate(zip(filtered_boxes, filtered_labels, filtered_scores)):
        x1, y1, x2, y2 = box
        class_name = class_names.get(int(label), f'Unknown_{int(label)}')

        detection_info = {
            'index': i,
            'box': [x1, y1, x2, y2],
            'label_id': int(label),
            'class_name': class_name,
            'score': float(score),
            'width': x2 - x1,
            'height': y2 - y1
        }

        filtered_detections.append(detection_info)

        print(f"Detection {i:3d}: {class_name:15s} | "
              f"Score: {score:.3f} | "
              f"Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | "
              f"Size: {x2-x1:.1f}x{y2-y1:.1f}")

    # 클래스별 통계
    print(f"\n{'='*60}")
    print("클래스별 Detection 개수:")
    print(f"{'='*60}")

    class_counts = {}
    for det in filtered_detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:15s}: {count:3d}개")

    return filtered_detections


def visualize_detections(image, detections, class_names, score_threshold=0.5, save_path=None):
    """
    Detection 결과를 이미지에 시각화합니다.

    Args:
        image: PIL.Image or numpy array or torch.Tensor
        detections: Dict with 'boxes', 'labels', 'scores'
        class_names: Dict mapping label ID to class name
        score_threshold: 최소 confidence 점수
        save_path: 저장 경로 (optional)
    """
    # 이미지를 numpy array로 변환
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.min() < 0:  # Normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    # Detection 필터링
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()

    mask = scores >= score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_np)
    ax.set_title(f"Detections (threshold={score_threshold}, count={len(boxes)})")

    # 클래스별 색상
    colors = {
        'Vehicles': 'red',
        'Pedestrians': 'blue',
        'Cyclists': 'green',
        'Signs': 'yellow',
        'person': 'blue',
        'car': 'red',
        'truck': 'orange',
        'bus': 'purple'
    }
    default_color = 'cyan'

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        class_name = class_names.get(int(label), f'Class_{int(label)}')
        color = colors.get(class_name, default_color)

        # 박스 그리기
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # 라벨 텍스트
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 시각화 저장: {save_path}")

    return fig


def analyze_score_distribution(detections):
    """
    Score 분포를 분석합니다.
    """
    scores = detections['scores'].cpu().numpy()

    print(f"\n{'='*60}")
    print("Score 분포 분석")
    print(f"{'='*60}")

    thresholds = [0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        count = (scores >= thresh).sum()
        percentage = 100 * count / len(scores)
        print(f"Score >= {thresh}: {count:4d}개 ({percentage:5.1f}%)")

    # Score 히스토그램
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Detection Score Distribution')
    ax.grid(True, alpha=0.3)

    # Threshold 선 표시
    for thresh in [0.5, 0.7, 0.9]:
        ax.axvline(thresh, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold {thresh}')
    ax.legend()

    return fig


def example_usage():
    """사용 예제"""
    import sys
    import os

    PROJECT_NAME = "customRCNN"
    PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
    DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

    sys.path.insert(0, DETECTION_DIR)
    sys.path.insert(0, PROJECT_DIR)

    from DeepDataMiningLearning.detection.models import create_detectionmodel

    # 모델 생성
    print("모델 로딩 중...")
    model, _, _ = create_detectionmodel(
        modelname='fasterrcnn_resnet50_fpn_v2',
        num_classes=5,  # Waymo: background + 4 classes
        trainable_layers=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.eval()

    # 테스트 이미지 생성
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_image = torch.rand(3, 800, 1200).to(device)

    # Inference
    print("\nInference 실행 중...")
    with torch.no_grad():
        detections = model([test_image])

    detection = detections[0]  # 첫 번째 이미지

    # 1. Detection 해석
    filtered_dets = interpret_detection(detection, WAYMO_CLASSES, score_threshold=0.5)

    # 2. Score 분포 분석
    score_fig = analyze_score_distribution(detection)
    score_fig.savefig('score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(score_fig)

    # 3. 시각화
    vis_fig = visualize_detections(
        test_image,
        detection,
        WAYMO_CLASSES,
        score_threshold=0.5,
        save_path='detection_result.png'
    )
    plt.close(vis_fig)

    print("\n✅ 완료!")


if __name__ == "__main__":
    example_usage()
