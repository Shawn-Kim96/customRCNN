"""
Train Advanced Backbone R-CNN

ViT, Swin Transformer를 backbone으로 사용한 detection model 학습
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path

PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.modeling.advanced_backbone_rcnn import AdvancedBackboneRCNN
from DeepDataMiningLearning.detection.dataset_waymococo import WaymoCOCODataset, get_transformsimple
from DeepDataMiningLearning.detection.dataset_nuscenescoco import NuscenesCOCODataset
from DeepDataMiningLearning.detection.dataset_nuscenes import create_nuscenes_transforms


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    model.train()
    total_losses = {}

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}/{args.epochs}')

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        total_loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v.item()

        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in losses.items()})

    avg_losses = {k: v / len(data_loader) for k, v in total_losses.items()}
    print(f"\nEpoch {epoch} Losses:", {k: f'{v:.4f}' for k, v in avg_losses.items()})

    return avg_losses


def train(model, optimizer_param, scheduler_param, model_save_path, data_loader, device, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, **optimizer_param)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)

    # Output dir
    output_dir = Path(model_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, optimizer, data_loader, device, epoch, args)
        lr_scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save final
    torch.save({
        'model': model.state_dict(),
        'args': args
    }, output_dir / 'final_model.pth')

    print(f"\nTraining completed! Models saved in {output_dir}")
