"""
COCO Evaluation for Object Detection Models

Provides clean COCO mAP evaluation using pycocotools
"""

import torch
import sys
import os
from tqdm import tqdm

PROJECT_NAME = "customRCNN"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DETECTION_DIR = os.path.join(PROJECT_DIR, "DeepDataMiningLearning", "detection")

sys.path.insert(0, DETECTION_DIR)
sys.path.insert(0, PROJECT_DIR)

from DeepDataMiningLearning.detection.myevaluator import (
    CocoEvaluator,
    get_coco_api_from_dataset
)


@torch.inference_mode()
def evaluate_coco(model, data_loader, device):
    """
    Evaluate object detection model using COCO metrics.

    Args:
        model: Detection model
        data_loader: Test/Val dataloader
        device: torch.device

    Returns:
        CocoEvaluator with results

    Prints:
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
    """
    cpu_device = torch.device("cpu")
    model.eval()

    # Convert dataset to COCO format
    print("Converting dataset to COCO format...")
    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = ["bbox"]
    coco.dataset['info'] = {}
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("Running evaluation...")
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]

        # Model inference
        outputs = model(images)

        # Move to CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # Create result dict: {image_id: prediction}
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

    # Accumulate and summarize
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def example_usage():
    """Example: How to use evaluate_coco"""
    from torch.utils.data import DataLoader
    from DeepDataMiningLearning.detection.dataset_waymococo import WaymoCOCODataset, get_transformsimple
    from src.modeling.advanced_backbone_rcnn import AdvancedBackboneRCNN

    # Load dataset
    data_root = '/data/Datasets/WaymoCOCO/Training'
    ann_file = os.path.join(data_root, 'annotations.json')

    test_dataset = WaymoCOCODataset(
        root=data_root,
        annotation=ann_file,
        train=False,
        transform=get_transformsimple(train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedBackboneRCNN(
        backbone_name='vit_b_16',
        num_classes=5,
        pretrained=True
    ).to(device)

    # Evaluate
    evaluate_coco(model, test_loader, device)


if __name__ == "__main__":
    example_usage()
