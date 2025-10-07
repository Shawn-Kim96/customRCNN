import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import sys
import os
from pathlib import Path

PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PROJECT_DIR)

print("Downloading ViT-B/16 weights...")
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Save to file
model_save_dir = Path(os.path.join(PROJECT_DIR, 'data', 'pretrained_models', 'vit'))
model_save_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), f'{model_save_dir}/vit_b_16_weights.pth')
print(f"Saved to: {model_save_dir}/vit_b_16_weights.pth")
print(f"   Size: {os.path.getsize(f'{model_save_dir}/vit_b_16_weights.pth') / 1e6:.1f} MB")