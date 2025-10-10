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

from src.tools.evaluate import evaluate_coco

# Backbone registries shared with training scripts.
# Advanced variants rely on transformer-based feature extractors,
# while standard variants fall back to torchvision ResNet + FPN.
ADVANCED_BACKBONES = {
    "vit_b_16": {"type": "vit"},
    "vit_b_32": {"type": "vit"},
    "vit_l_16": {"type": "vit"},
    "swin_t": {"type": "swin"},
    "swin_s": {"type": "swin"},
    "swin_b": {"type": "swin"},
}

STANDARD_BACKBONES = {
    "resnet50": {"trainable_layers": 2},
    "resnet101": {"trainable_layers": 2},
    "resnet152": {"trainable_layers": 2},
}

BACKBONE_CHOICES = tuple(list(ADVANCED_BACKBONES.keys()) + list(STANDARD_BACKBONES.keys()))


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


def train(model, optimizer_param, scheduler_param, model_save_path, train_loader, val_loader, device, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, **optimizer_param)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)

    # Output dir
    output_dir = Path(model_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import evaluation function
    from src.tools.evaluate import evaluate_coco

    # Training
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, args)
        lr_scheduler.step()

        # Validation evaluation (every epoch or every N epochs)
        if val_loader is not None and (epoch + 1) % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Validation @ Epoch {epoch+1}")
            print(f"{'='*60}")
            evaluate_coco(model, val_loader, device)

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


def train_combined(model, optimizer_param, scheduler_param, model_save_path,
                    train_loader, val_waymo_loader, val_nuscenes_loader, device, args):
    """
    Train on combined dataset, validate on each dataset separately

    Args:
        model: Detection model
        optimizer_param: Optimizer parameters (lr, momentum, weight_decay)
        scheduler_param: LR scheduler parameters (step_size, gamma)
        model_save_path: Directory to save checkpoints
        train_loader: Combined train dataloader (Waymo + Nuscenes)
        val_waymo_loader: Waymo validation dataloader
        val_nuscenes_loader: Nuscenes validation dataloader
        device: torch.device
        args: Training arguments
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, **optimizer_param)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)

    # Output dir
    output_dir = Path(model_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    model_variant = getattr(args, "model_variant", "advanced" if getattr(args, "backbone", None) in ADVANCED_BACKBONES else "standard")
    print(f"Model variant: {model_variant}")
    if hasattr(args, "backbone"):
        print(f"Backbone: {args.backbone}")
    print(f"Total epochs: {args.epochs}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val Waymo samples: {len(val_waymo_loader.dataset)}")
    print(f"Val Nuscenes samples: {len(val_nuscenes_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Evaluation interval: {args.eval_interval} epochs")
    print(f"{'='*70}\n")

    # Training loop
    for epoch in range(args.epochs):
        # Train one epoch on combined dataset
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, args)
        lr_scheduler.step()

        # Validation on both datasets separately
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\n{'='*70}")
            print(f"Validation @ Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*70}")

            # Waymo validation
            print(f"\n--- Waymo Dataset Validation ---")
            evaluate_coco(model, val_waymo_loader, device)

            # Nuscenes validation
            print(f"\n--- Nuscenes Dataset Validation ---")
            evaluate_coco(model, val_nuscenes_loader, device)

            print(f"{'='*70}\n")

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'model': model.state_dict(),
        'args': args
    }, final_model_path)

    print(f"\nTraining completed! Final model saved: {final_model_path}")
