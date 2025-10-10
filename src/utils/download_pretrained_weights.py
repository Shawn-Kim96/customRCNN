import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torchvision.models import (
    swin_b,
    swin_s,
    swin_t,
    vit_b_16,
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
)

PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath(".").split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PROJECT_DIR)


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "vit_b_16": {
        "constructor": vit_b_16,
        "weights": ViT_B_16_Weights.IMAGENET1K_V1,
        "subdir": "vit",
        "filename": "vit_b_16_weights.pth",
    },
    "swin_t": {
        "constructor": swin_t,
        "weights": Swin_T_Weights.IMAGENET1K_V1,
        "subdir": "swin",
        "filename": "swin_t_weights.pth",
    },
    "swin_s": {
        "constructor": swin_s,
        "weights": Swin_S_Weights.IMAGENET1K_V1,
        "subdir": "swin",
        "filename": "swin_s_weights.pth",
    },
    "swin_b": {
        "constructor": swin_b,
        "weights": Swin_B_Weights.IMAGENET1K_V1,
        "subdir": "swin",
        "filename": "swin_b_weights.pth",
    },
}


def download_and_save(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")

    entry = MODEL_REGISTRY[model_name]
    constructor = entry["constructor"]
    weights = entry["weights"]
    subdir = entry["subdir"]
    filename = entry["filename"]

    print(f"\nDownloading weights for {model_name}...")
    model = constructor(weights=weights)

    save_dir = Path(PROJECT_DIR) / "data" / "pretrained_models" / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"Saved to: {save_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights for ViT and Swin backbones.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help=f"Models to download. Supported: {', '.join(MODEL_REGISTRY.keys())}.",
    )
    args = parser.parse_args()

    for model_name in args.models:
        download_and_save(model_name)


if __name__ == "__main__":
    main()
