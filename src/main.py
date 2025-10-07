"""
Main Training Script for Advanced Backbone R-CNN

Trains on combined Waymo + Nuscenes datasets
Evaluates separately on each dataset
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import sys
import os
import argparse
from pathlib import Path

PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.modeling.advanced_backbone_rcnn import AdvancedBackboneRCNN
from src.dataset.load_dataset import load_nuscenes_dataset, load_waymo_dataset
from src.tools.train_advanced import train_combined
from src.tools.evaluate import evaluate_coco


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train Advanced Backbone R-CNN on Combined Datasets')

    # Dataset - Waymo
    parser.add_argument('--waymo-root', default='data/waymo',
                        help='Waymo dataset root path')
    parser.add_argument('--waymo-annotation', default='annotations.json',
                        help='Waymo annotation file name')

    # Dataset - Nuscenes
    parser.add_argument('--nuscenes-root', default='data/nuscenes',
                        help='Nuscenes dataset root path')
    parser.add_argument('--nuscenes-annotation', default='annotations.json',
                        help='Nuscenes annotation file name')

    # Dataset split ratios
    parser.add_argument('--train-ratio', default=0.8, type=float,
                        help='train dataset ratio')
    parser.add_argument('--val-ratio', default=0.1, type=float,
                        help='validation dataset ratio')
    parser.add_argument('--random-seed', default=42, type=int,
                        help='random seed for dataset split')

    # Model
    parser.add_argument('--backbone', default='vit_b_16',
                        choices=['vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_t', 'swin_s', 'swin_b'],
                        help='backbone architecture')
    parser.add_argument('--num-classes', default=5, type=int,
                        help='number of classes (background + 4 classes)')
    parser.add_argument('--pretrained', default=True, type=bool,
                        help='use pretrained backbone (bool)')
    parser.add_argument('--pretrained-path', default='data/pretrained_models/vit/vit_b_16_weights.pth',
                        help='pretrained weight path')

    # Training
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch size for training')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--lr-step-size', default=5, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--eval-interval', default=2, type=int,
                        help='run validation every N epochs')
    parser.add_argument('--checkpoint-interval', default=5, type=int,
                        help='save checkpoint every N epochs')

    # Misc
    parser.add_argument('--output-dir', default='src/result',
                        help='path to save outputs')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--device', default='cuda', help='device (cuda or cpu)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================
    # Load Datasets using load_dataset.py
    # ========================================
    print("\n" + "="*70)
    print("Loading Datasets...")
    print("="*70)

    # Waymo datasets
    print("\n--- Waymo Dataset ---")
    waymo_ann_file = os.path.join(args.waymo_root, args.waymo_annotation)
    waymo_train, waymo_val, waymo_test = load_waymo_dataset(
        data_dir=args.waymo_root,
        anno_path=waymo_ann_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    print(f"Waymo Train: {len(waymo_train)} samples")
    print(f"Waymo Val: {len(waymo_val)} samples")
    print(f"Waymo Test: {len(waymo_test)} samples")

    # Nuscenes datasets
    print("\n--- Nuscenes Dataset ---")
    nuscenes_ann_file = os.path.join(args.nuscenes_root, args.nuscenes_annotation)
    nuscenes_train, nuscenes_val, nuscenes_test = load_nuscenes_dataset(
        data_dir=args.nuscenes_root,
        anno_path=nuscenes_ann_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    print(f"Nuscenes Train: {len(nuscenes_train)} samples")
    print(f"Nuscenes Val: {len(nuscenes_val)} samples")
    print(f"Nuscenes Test: {len(nuscenes_test)} samples")

    # Combine training datasets
    print("\n--- Combined Training Dataset ---")
    combined_train_dataset = ConcatDataset([waymo_train, nuscenes_train])
    print(f"Total Train: {len(combined_train_dataset)} samples "
          f"(Waymo: {len(waymo_train)}, Nuscenes: {len(nuscenes_train)})")

    # ========================================
    # Create DataLoaders
    # ========================================
    print("\nCreating DataLoaders...")

    # Combined training loader
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    # Separate validation loaders
    val_waymo_loader = DataLoader(
        waymo_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    val_nuscenes_loader = DataLoader(
        nuscenes_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    # Test loaders for final evaluation
    test_waymo_loader = DataLoader(
        waymo_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    test_nuscenes_loader = DataLoader(
        nuscenes_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    # ========================================
    # Create Model
    # ========================================
    print(f"\n{'='*70}")
    print(f"Creating Model: {args.backbone}")
    print(f"{'='*70}")

    model = AdvancedBackboneRCNN(
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        weights_path=args.pretrained_path
    ).to(device)

    print(f"Model created with {args.num_classes} classes")
    print(f"Pretrained backbone: {args.pretrained}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")

    # ========================================
    # Optimizer & Scheduler Parameters
    # ========================================
    optimizer_param = {
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }

    scheduler_param = {
        'step_size': args.lr_step_size,
        'gamma': args.lr_gamma
    }

    # ========================================
    # Training
    # ========================================
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}\n")

    output_dir = Path(os.path.join(Path(args.output_dir), f'advanced_backbone_{args.backbone}', f'ep{args.epochs}_lr{args.lr}_mom{args.momentum}_wd{args.weight_decay}'))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_combined(
        model=model,
        optimizer_param=optimizer_param,
        scheduler_param=scheduler_param,
        model_save_path=output_dir,
        train_loader=train_loader,
        val_waymo_loader=val_waymo_loader,
        val_nuscenes_loader=val_nuscenes_loader,
        device=device,
        args=args
    )

    # ========================================
    # Final Evaluation on Test Sets (Both Datasets)
    # ========================================
    print(f"\n{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print("Waymo Dataset - Final Test Results")
    print(f"{'='*70}")
    evaluate_coco(model, test_waymo_loader, device)

    print(f"\n{'='*70}")
    print("Nuscenes Dataset - Final Test Results")
    print(f"{'='*70}")
    evaluate_coco(model, test_nuscenes_loader, device)

    print(f"\n{'='*70}")
    print("Training and Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Models saved in: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
