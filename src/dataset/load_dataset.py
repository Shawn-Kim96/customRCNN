import torch
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import random_split


PROJECT_NAME = "customRCNN"
PORJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PORJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PORJECT_DIR)

from dataset_nuscenescoco import NuscenesCOCODataset
from dataset_nuscenes import create_nuscenes_transforms
from dataset_waymococo import WaymoCOCODataset, get_transformsimple
from src.utils.visualize import visualize_dataset_sample


NUSCENES_CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'trailer',
    4: 'construction_vehicle',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'traffic_cone',
    9: 'barrier',
}

WAYMO_CLASS_NAMES = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
    4: "Sign"
}


def load_nuscenes_dataset(data_dir, anno_path, train_ratio: float = 0.8, val_ratio: float = 0.1, random_seed: int=42):
    transform = create_nuscenes_transforms(train=True)

    dataset = NuscenesCOCODataset(
        root=data_dir,
        annotation=anno_path,
        train=True,
        transform=transform
    )

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # torch.save(train_dataset, os.path.join(dataset_dir, 'train_nuscenes_dataset.pt'))
    # torch.save(val_dataset, os.path.join(dataset_dir, 'val_nuscenes_dataset.pt'))
    # torch.save(test_dataset, os.path.join(dataset_dir, 'test_nuscenes_dataset.pt'))
    return train_dataset, val_dataset, test_dataset


def load_waymo_dataset(data_dir, anno_path, train_ratio: float = 0.8, val_ratio: float = 0.1, random_seed: int=42):
    transform = get_transformsimple(None)
    dataset = WaymoCOCODataset(
        root=data_dir,
        annotation=anno_path,
        train=True,
        transform=transform
    )

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size # Remaining for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # torch.save(train_dataset, os.path.join(dataset_dir, 'train_nuscenes_dataset.pt'))
    # torch.save(val_dataset, os.path.join(dataset_dir, 'val_nuscenes_dataset.pt'))
    # torch.save(test_dataset, os.path.join(dataset_dir, 'test_nuscenes_dataset.pt'))
    return train_dataset, val_dataset, test_dataset


def test_nuscenes_dataset():
    transform = create_nuscenes_transforms(train=True)
    dataset = NuscenesCOCODataset(
        root='data/nuscenes_subset_coco_step10',
        annotation='data/nuscenes_subset_coco_step10/annotations.json',
        train=True,
        transform=transform
    )

    num_samples = min(5, len(dataset))

    for idx in range(num_samples):
        image, target = dataset[idx]
        print(f"\nImage shape: {image.shape}")
        print(f"Number of boxes: {len(target['boxes'])}")
        print(f"Labels: {target['labels']}")

        fig = visualize_dataset_sample(image, target, idx, NUSCENES_CLASS_NAMES)

        output_path = f'src/dataset/nuscene_dataset_sample_{idx}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"image saved: {output_path}")

        plt.close(fig)


def test_waymo_dataset():
    transform = get_transformsimple(None)
    dataset = WaymoCOCODataset(
        root='data/waymo',
        annotation = 'data/waymo/annotations.json',
        train=True,
        transform=transform
    )

    num_samples = min(5, len(dataset))

    for idx in range(num_samples):
        image, target = dataset[idx]
        print(type(image))
        print(f"\nImage shape: {image.shape}")
        print(f"Number of boxes: {len(target['boxes'])}")
        print(f"Labels: {target['labels']}")

        fig = visualize_dataset_sample(image, target, idx, WAYMO_CLASS_NAMES)

        output_path = f'src/dataset/waymo_dataset_sample_{idx}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"image saved: {output_path}")

        plt.close(fig)



if __name__ == "__main__":
    test_nuscenes_dataset()
    # test_waymo_dataset()