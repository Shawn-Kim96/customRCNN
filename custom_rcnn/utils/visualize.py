import torch
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_dataset_sample(image, target, index, class_names):
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()

        if img_np.min() < 0:  # Normalized된 경우
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.array(image)

    img_rgb = img_np

    h, w = img_rgb.shape[:2]

    boxes = target['boxes']
    labels = target['labels']

    print(f"\n========== Sample {index} ==========")
    print(f"image shape: {img_rgb.shape}")
    print(f"Bounding boxes cnt: {len(boxes)}")

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.item()

        xmin, ymin, xmax, ymax = box
        class_name = class_names.get(int(label), f'Class_{int(label)}')
        box_w = xmax - xmin
        box_h = ymax - ymin

        print(f"  Box {i}: class={int(label)} ({class_name}), "
              f"bbox(xyxy)=[{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}], "
              f"size=[{box_w:.1f} x {box_h:.1f}]")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(f"Sample {index} - {len(boxes)} objects")

    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.item()

        xmin, ymin, xmax, ymax = box
        box_w = xmax - xmin
        box_h = ymax - ymin

        class_name = class_names.get(int(label), f'Class_{int(label)}')
        color = colors[int(label) % len(colors)]

        rect = patches.Rectangle((xmin, ymin), box_w, box_h,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        ax.text(xmin, ymin-5, f"{class_name}",
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontsize=10, color='black', weight='bold')

    ax.axis('off')
    plt.tight_layout()

    return fig