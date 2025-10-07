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


def get_args():
    parser = argparse.ArgumentParser()

    # Backbone
    parser.add_argument('--backbone-type', default='vit',
                       choices=['vit', 'swin', 'resnet50', 'resnet101', 'resnet152'],
                       help='Backbone type (vit works on older torchvision)')
    parser.add_argument('--backbone-name', default='vit_b_16')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights (requires internet connection)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from scratch (no internet required)')
    parser.add_argument('--pretrained-path', default=None, type=str,
                       help='Path to local pretrained weights (for offline use)')
    parser.set_defaults(pretrained=True)

    # Model
    parser.add_argument('--num-classes', default=5, type=int)

    # Dataset
    parser.add_argument('--data-path', default='data/waymo')
    parser.add_argument('--annotation', default='data/waymo/annotations.json')

    # Training
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--workers', default=4, type=int)

    # Quick test
    parser.add_argument('--subset-size', default=None, type=int,
                       help='Use only N samples for quick testing (e.g., 20)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: subset-size=20, epochs=5, batch-size=2')

    # Output
    parser.add_argument('--output-dir', default='src/result')
    parser.add_argument('--resume', default='')

    return parser.parse_args()


def collate_fn(batch):
    return tuple(zip(*batch))


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


def main():
    args = get_args()

    # Quick test mode
    if args.quick_test:
        print("\nQuick Test Mode Enabled!")
        args.subset_size = 20
        args.epochs = 5
        args.batch_size = 2
        print(f"   Subset: {args.subset_size} samples")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Training Advanced Backbone R-CNN")
    print(f"  Device: {device}")
    print(f"  Backbone: {args.backbone_type}")
    if args.subset_size:
        print(f"  Subset size: {args.subset_size} samples")
    print(f"{'='*60}\n")

    # Dataset
    from DeepDataMiningLearning.detection import transforms as T
    from torch.utils.data import Subset

    def get_transform(train):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ToDtype(torch.float, scale=True))
        return T.Compose(transforms)

    full_dataset = WaymoCOCODataset(
        root=args.data_path,
        annotation=args.annotation,
        train=True,
        transform=get_transform(True)
    )

    # Apply subset if specified
    if args.subset_size:
        import random
        indices = random.sample(range(len(full_dataset)), min(args.subset_size, len(full_dataset)))
        train_dataset = Subset(full_dataset, indices)
        print(f"Using subset: {len(train_dataset)} / {len(full_dataset)} samples")
    else:
        train_dataset = full_dataset

    # do not include background
    label_dict = {i+1: value for i, value in enumerate(full_dataset.INSTANCE_CATEGORY_NAMES[1:])}
    print(f"Dataset: {len(train_dataset)} images\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    # Model
    print("Creating model...")
    model = AdvancedBackboneRCNN(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        weights_path=args.pretrained_path,
        num_classes=args.num_classes
    )
    model = model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Output dir
    output_dir = Path(os.path.join(Path(args.output_dir), f'advanced_backbone_{args.backbone_type}', f'ep{args.epochs}_lr{args.lr}'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, args)
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


if __name__ == '__main__':
    main()
