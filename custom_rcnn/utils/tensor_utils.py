"""
Tensor 변환 유틸리티 함수들
"""

import torch
import numpy as np
from typing import Dict, List, Any, Union


def detections_to_numpy(detections: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """
    Detection 결과의 모든 tensor를 numpy array로 변환합니다.

    Args:
        detections: Dict 또는 List[Dict] 형태의 detection 결과
                   각 dict는 'boxes', 'labels', 'scores' 등의 tensor를 포함

    Returns:
        numpy array로 변환된 detection 결과 (같은 구조)

    Examples:
        >>> # 단일 detection
        >>> detection = {'boxes': tensor([...]), 'labels': tensor([...]), 'scores': tensor([...])}
        >>> numpy_det = detections_to_numpy(detection)
        >>> type(numpy_det['boxes'])  # numpy.ndarray

        >>> # 여러 이미지의 detections
        >>> detections = [detection1, detection2, ...]
        >>> numpy_dets = detections_to_numpy(detections)
    """
    def _to_numpy(obj):
        """재귀적으로 tensor를 numpy로 변환"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            return {k: _to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_numpy(item) for item in obj)
        else:
            return obj

    return _to_numpy(detections)


def detections_to_cpu(detections: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """
    Detection 결과의 모든 tensor를 CPU로 이동합니다.

    Args:
        detections: Dict 또는 List[Dict] 형태의 detection 결과

    Returns:
        CPU로 이동된 detection 결과
    """
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_cpu(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_cpu(item) for item in obj)
        else:
            return obj

    return _to_cpu(detections)


def dict_apply(data: Dict, func) -> Dict:
    """
    딕셔너리의 모든 값에 함수를 적용합니다.

    Args:
        data: 변환할 딕셔너리
        func: 적용할 함수 (예: lambda x: x.detach().numpy())

    Returns:
        변환된 딕셔너리

    Examples:
        >>> detection = {'boxes': tensor([...]), 'labels': tensor([...])}
        >>> numpy_det = dict_apply(detection, lambda x: x.detach().cpu().numpy())
    """
    return {k: func(v) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}


def batch_detections_to_numpy(detections_list: List[Dict]) -> List[Dict]:
    """
    배치 detection 결과를 numpy로 변환합니다.

    Args:
        detections_list: List[Dict] 형태의 detection 결과들

    Returns:
        numpy로 변환된 detection 결과들
    """
    return [detections_to_numpy(det) for det in detections_list]


# 편의성을 위한 짧은 alias
to_numpy = detections_to_numpy
to_cpu = detections_to_cpu


if __name__ == "__main__":
    # 테스트 코드
    print("=== Tensor Utils 테스트 ===\n")

    # 테스트 데이터 생성
    dummy_detection = {
        'boxes': torch.rand(100, 4),
        'labels': torch.randint(0, 5, (100,)),
        'scores': torch.rand(100)
    }

    print("원본 타입:")
    for k, v in dummy_detection.items():
        print(f"  {k}: {type(v)} {v.shape}")

    # Numpy로 변환
    numpy_detection = detections_to_numpy(dummy_detection)

    print("\nNumpy 변환 후:")
    for k, v in numpy_detection.items():
        print(f"  {k}: {type(v)} {v.shape}")

    # 배치 테스트
    batch_detections = [dummy_detection.copy() for _ in range(3)]
    numpy_batch = batch_detections_to_numpy(batch_detections)

    print(f"\n배치 변환: {len(batch_detections)}개 → {len(numpy_batch)}개")
    print(f"첫 번째 detection boxes 타입: {type(numpy_batch[0]['boxes'])}")

    print("\n✅ 모든 테스트 통과!")
