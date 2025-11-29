"""
Advanced Backbone R-CNN

R-CNN using pre-trained advanced backbones (DINO, Swin Transformer) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import sys
import os
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    swin_t,
    Swin_T_Weights,
    swin_s,
    Swin_S_Weights,
    swin_b,
    Swin_B_Weights,
)


PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.insert(0, os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection"))
sys.path.insert(0, PROJECT_DIR)

from src.modeling.modeling_rpnfasterrcnn import CustomRCNN


class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT) Backbone with Adaptive Position Embedding
    Supports arbitrary input sizes by interpolating position embeddings
    """
    def __init__(self, model_name='vit_b_16', pretrained=True, weights_path=None):
        """
        Args:
            model_name: ViT model name
            pretrained: Use pretrained weights
            weights_path: Path to local weights file (for offline use)
        """
        super().__init__()

        # Load model with weights
        if pretrained and weights_path:
            # Load from local file (HPC offline mode)
            print(f"Loading ViT weights from: {weights_path}")
            self.vit = vit_b_16(weights=None)  # Create architecture only
            state_dict = torch.load(weights_path, map_location='cpu')
            self.vit.load_state_dict(state_dict)
            print("ViT weights loaded successfully!")
        elif pretrained:
            # Download from internet (requires connection)
            print("Downloading ViT pretrained weights...")
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            # Random initialization
            print("Using random initialization (no pretrained weights)")
            self.vit = vit_b_16(weights=None)

        # ViT-B/16: 768 channels
        self.out_channels = 768
        self.patch_size = 16

    def interpolate_pos_embedding(self, pos_embed, h, w):
        """Interpolate position embeddings for arbitrary input sizes"""
        N = pos_embed.shape[1] - 1  # Exclude cls token
        D = pos_embed.shape[2]

        if h * w == N:
            return pos_embed

        # Separate class token and patch embeddings
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]

        # Reshape and interpolate
        old_h = old_w = int(N ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, D)

        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: OrderedDict {'0': [B, 768, H//16, W//16]}
        """
        B, C, H, W = x.shape

        # Patchify
        x = self.vit.conv_proj(x)  # [B, 768, h, w]
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, h*w, 768]

        # Add class token
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, h*w+1, 768]

        # Interpolate position embedding
        pos_embed = self.interpolate_pos_embedding(self.vit.encoder.pos_embedding, h, w)
        x = x + pos_embed

        # Apply transformer
        x = self.vit.encoder.dropout(x)
        for layer in self.vit.encoder.layers:
            x = layer(x)
        x = self.vit.encoder.ln(x)

        # Remove cls token and reshape
        x = x[:, 1:]  # [B, h*w, 768]
        x = x.transpose(1, 2).reshape(B, self.out_channels, h, w)

        return OrderedDict([('0', x)])


class SwinBackbone(nn.Module):
    """
    Swin Transformer Backbone

    Hierarchical vision transformer, better for detection tasks.
    """
    def __init__(self, model_name='swin_t', pretrained=True, weights_path: Optional[str] = None):
        super().__init__()

        constructors = {
            'swin_t': (swin_t, Swin_T_Weights),
            'swin_s': (swin_s, Swin_S_Weights),
            'swin_b': (swin_b, Swin_B_Weights),
        }
        if model_name not in constructors:
            raise ValueError(f"Unsupported Swin model: {model_name}")

        constructor, weight_enum = constructors[model_name]

        if pretrained and weights_path:
            print(f"Loading Swin weights from: {weights_path}")
            self.swin = constructor(weights=None)
            state_dict = torch.load(weights_path, map_location='cpu')
            self.swin.load_state_dict(state_dict)
            print("Swin weights loaded successfully!")
        elif pretrained:
            print(f"Downloading Swin pretrained weights ({model_name})...")
            self.swin = constructor(weights=weight_enum.IMAGENET1K_V1)
        else:
            print(f"Using random initialization for Swin backbone: {model_name}")
            self.swin = constructor(weights=None)

        # Stage-4 channel dimension depends on model size
        self.out_channels = self.swin.head.in_features

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            features: OrderedDict
        """
        # Swin features (only use last stage for simplicity)
        features = self.swin.features(x)  # [B, H//32, W//32, 768]
        features = features.permute(0, 3, 1, 2).contiguous()  # [B, 768, H//32, W//32]

        # Optionally upsample for higher resolution
        features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=True)
        # features: [B, 768, H//16, W//16]

        return OrderedDict([('0', features)])


class SimpleFPN(nn.Module):
    """
    Simplified FPN for advanced backbones

    단일 feature map을 multi-scale로 변환합니다.
    """
    def __init__(self, in_channels=768, out_channels=256):
        super().__init__()

        # Lateral connections
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)

        # Top-down pathway with upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Downsampling for additional levels
        self.downsample1 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            features: OrderedDict with keys '0', '1', '2', '3', 'pool'
        """
        # Lateral connection
        p = self.lateral(x)  # [B, 256, H, W]

        # Create multi-scale features (5 levels to match RPN anchors)
        # P3 (stride=8, assuming input is stride=16)
        p3 = self.upsample(p)  # [B, 256, H*2, W*2]
        p3 = self.smooth1(p3)

        # P4 (stride=16)
        p4 = p  # [B, 256, H, W]

        # P5 (stride=32)
        p5 = self.downsample1(p4)  # [B, 256, H/2, W/2]
        p5 = self.smooth2(p5)

        # P6 (stride=64)
        p6 = self.downsample2(p5)  # [B, 256, H/4, W/4]

        # P7/pool (stride=128)
        p7 = self.downsample3(p6)  # [B, 256, H/8, W/8]
        p7 = self.smooth3(p7)

        return OrderedDict([
            ('0', p3),
            ('1', p4),
            ('2', p5),
            ('3', p6),
            ('pool', p7)
        ])


class AdvancedBackboneRCNN(CustomRCNN):
    """
    CustomRCNN with Advanced Backbone

    ViT, Swin Transformer 등 advanced backbone을 사용합니다.
    """
    def __init__(
        self,
        backbone_type='vit',  # 'vit', 'swin', 'dino'
        backbone_name='vit_b_16',
        pretrained=True,
        num_classes=5,
        out_channels=256,
        **kwargs
    ):
        # Don't call super().__init__ yet - we need to set up backbone first

        # Import base class components but don't initialize
        from torch import nn
        nn.Module.__init__(self)

        print(f"\n{'='*60}")
        print(f"Initializing AdvancedBackboneRCNN")
        print(f"  Backbone type: {backbone_type}")
        print(f"  Backbone name: {backbone_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"{'='*60}\n")

        # Create advanced backbone
        if backbone_type == 'vit':
            self.backbone_net = ViTBackbone(
                model_name=backbone_name,
                pretrained=pretrained,
                weights_path=kwargs.get('weights_path', None)
            )
        elif backbone_type == 'swin':
            self.backbone_net = SwinBackbone(
                model_name=backbone_name,
                pretrained=pretrained,
                weights_path=kwargs.get('weights_path', None)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Add FPN on top
        self.fpn = SimpleFPN(
            in_channels=self.backbone_net.out_channels,
            out_channels=out_channels
        )

        # Set out_channels for RPN/ROI heads
        self.out_channels = out_channels

        # Now initialize the rest of CustomRCNN components
        # (Transform, RPN, ROI heads)
        from DeepDataMiningLearning.detection.detectiontransform import DetectionTransform
        from src.modeling.modeling_rpnfasterrcnn import (
            AnchorGenerator, RPNHead, RegionProposalNetwork,
            MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor, RoIHeads
        )

        # Transform
        min_size = kwargs.get('min_size', 800)
        max_size = kwargs.get('max_size', 1333)
        image_mean = kwargs.get('image_mean', [0.485, 0.456, 0.406])
        image_std = kwargs.get('image_std', [0.229, 0.224, 0.225])

        self.detcttransform = DetectionTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            size_divisible=32
        )

        # RPN
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(self.out_channels, self.rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = {'training': 2000, 'testing': 1000}
        rpn_post_nms_top_n = {'training': 2000, 'testing': 1000}

        self.rpn = RegionProposalNetwork(
            self.rpn_anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh=0.7,
            score_thresh=0.0
        )

        # ROI Heads
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2
        )

        representation_size = 1024
        box_head = TwoMLPHead(self.out_channels * 7 * 7, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

        print(f"✅ AdvancedBackboneRCNN initialized successfully!")

    def forward(self, images, targets=None):
        """Forward pass (same as CustomRCNN but with advanced backbone)"""
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Preprocessing
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.detcttransform(images, targets)

        # Advanced backbone
        backbone_features = self.backbone_net(images.tensors)  # {'0': [B, 768, H, W]}

        # FPN
        features = self.fpn(backbone_features['0'])  # Multi-scale features

        # RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # ROI Heads
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # Post-process
        detections = self.detcttransform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        return detections


def test_advanced_backbone():
    """Test AdvancedBackboneRCNN"""
    print("\n" + "="*60)
    print("Testing AdvancedBackboneRCNN")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = AdvancedBackboneRCNN(
        backbone_type='vit',
        backbone_name='vit_b_16',
        pretrained=True,
        num_classes=5
    )
    model = model.to(device)

    # Test data
    batch_size = 2
    images = [torch.rand(3, 800, 1200).to(device) for _ in range(batch_size)]

    targets = []
    for i in range(batch_size):
        target = {
            'boxes': torch.rand(5, 4).to(device) * 800,
            'labels': torch.randint(1, 5, (5,)).to(device)
        }
        target['boxes'][:, 2:] = target['boxes'][:, :2] + target['boxes'][:, 2:]
        targets.append(target)

    # Training
    print("\n[Training Mode]")
    model.train()
    losses = model(images, targets)

    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Evaluation
    print("\n[Evaluation Mode]")
    model.eval()
    with torch.no_grad():
        detections = model(images)

    print(f"\nDetections: {len(detections)} images")
    for i, det in enumerate(detections):
        print(f"  Image {i}:")
        print(f"    boxes: {det['boxes'].shape}")
        print(f"    labels: {det['labels'].shape}")
        print(f"    scores: {det['scores'].shape}")

    print("\n✅ Test passed!")


if __name__ == "__main__":
    test_advanced_backbone()
