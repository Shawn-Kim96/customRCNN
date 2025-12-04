# MMDet3D 평가 디버그 노트

이 노트는 Waymo-KITTI 변환 데이터를 mmdet3d가 기대하는 포맷으로 맞추면서, 왜 평가(mAP)가 안 나오는지 한 줄씩 확인하는 과정을 정리합니다. (실제 노트북으로 옮겨도 되고, 아래 명령을 셀로 실행해도 됩니다.)

## 1) 기본 환경/경로
```bash
PYTHON=/home/018219422/miniconda3/envs/det3d310/bin/python
WAYMO_ROOT=/home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/kitti_format
CFG=/home/018219422/customRCNN/3d_detection/checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py
CKPT=/home/018219422/customRCNN/3d_detection/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
```

## 2) 기존 info 구조 확인 (리스트인지 dict인지)
```bash
$PYTHON - <<'PY'
import pickle
src="/home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/kitti_format/kitti_infos_val.pkl"
infos=pickle.load(open(src,'rb'))
print(type(infos), len(infos))
print("keys", infos[0].keys())
print("point_cloud", infos[0]['point_cloud'])
print("annos keys", infos[0]['annos'].keys())
PY
```
*결과 해석*: `infos`가 `list`이고, 각 항목에 `point_cloud/annos` 등이 들어 있음. 최신 mmengine은 `ann_file`을 dict(`{'metainfo':..., 'data_list': [...]}`)로 기대함 → 리스트 그대로 주면 TypeError 발생.

## 3) mmengine이 기대하는 키 확인
```bash
$PYTHON - <<'PY'
import inspect, mmdet3d.datasets.det3d_dataset as dd
print("\\n".join(inspect.getsource(dd.Det3DDataset.parse_data_info).split("\\n")[:40]))
PY
```
*핵심*: `info['lidar_points']['lidar_path']`와 `num_pts_feats`가 필요. 원본 Waymo-KITTI info에는 `point_cloud/velodyne_path`만 있어서 KeyError 발생.

## 4) 최소 필드만 갖는 ann_file 만들기 (lidar_points 추가, GT는 비움)
```bash
$PYTHON - <<'PY'
import pickle, os
src="/home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/kitti_format/kitti_infos_val.pkl"
out="/home/018219422/customRCNN/3d_detection/cache/kitti_infos_val_mmdet3d_v2.pkl"
infos=pickle.load(open(src,'rb'))
converted=[]
for info in infos:
    pc=info.get('point_cloud',{})
    converted.append({
        'lidar_points': {
            'lidar_path': pc.get('velodyne_path',''),
            'num_pts_feats': pc.get('num_features',4)
        },
        'instances': [],           # GT 비움 → mAP 불가, 추론만 가능
        'timestamp': info.get('timestamp', None)
    })
wrapped={'metainfo': {'classes':['Car']}, 'data_list': converted}
os.makedirs(os.path.dirname(out), exist_ok=True)
pickle.dump(wrapped, open(out,'wb'))
print("saved", out, "len", len(converted))
PY
```
*결과 해석*: 이제 `lidar_points` 키가 존재 → KeyError 해결. 단, `instances`가 비어 있어 평가 지표(mAP 등)는 계산되지 않음.

## 5) 평가 스크립트에 주입할 cfg-options (경로 + 새로운 ann_file)
```bash
mim test mmdet3d \
  --config $CFG \
  --checkpoint $CKPT \
  --work-dir /home/018219422/customRCNN/3d_detection/results/waymo_pointpillars_val \
  --cfg-options \
    test_dataloader.dataset.data_root=$WAYMO_ROOT \
    test_dataloader.dataset.ann_file=/home/018219422/customRCNN/3d_detection/cache/kitti_infos_val_mmdet3d_v2.pkl \
    test_dataloader.dataset.data_prefix.pts=training/velodyne \
    test_evaluator.ann_file=/home/018219422/customRCNN/3d_detection/cache/kitti_infos_val_mmdet3d_v2.pkl \
    val_dataloader.dataset.data_root=$WAYMO_ROOT \
    val_dataloader.dataset.ann_file=/home/018219422/customRCNN/3d_detection/cache/kitti_infos_val_mmdet3d_v2.pkl \
    val_dataloader.dataset.data_prefix.pts=training/velodyne \
    train_dataloader.dataset.dataset.data_root=$WAYMO_ROOT \
    train_dataloader.dataset.dataset.data_prefix.pts=training/velodyne \
    test_dataloader.batch_size=1 \
    test_dataloader.num_workers=2
```
*기대 결과*: 추론은 실행되지만 `instances`가 비어 있으므로 mAP 등 정량 지표는 나오지 않음(이미 GT가 없는 데이터라 그렇다).

## 6) GT가 있어야 나오는 평가 지표
- `instances` 필드에 `bbox_3d`/`bbox_label_3d` 등이 채워져야 `parse_ann_info`가 `gt_bboxes_3d`, `gt_labels_3d`를 만들어 mAP 계산 가능.
- 현재 Waymo-KITTI pkl에는 `annos`가 있지만 최신 포맷(`instances`)으로 변환돼 있지 않음 → mAP 불가.
- 해결하려면:
  1) `annos`를 mmengine 포맷(`instances`)으로 매핑 (예: bbox → bbox_3d, name → bbox_label_3d 등) 후 새 ann_file 생성  
  2) 또는 mmdet3d `create_data.py waymo ...`를 최신 버전으로 다시 실행해 포맷을 재생성

## 7) 추가 확인 포인트
- `data_prefix.pts`는 Waymo-KITTI에서는 `training/velodyne`로 맞춰야 함.
- GPU 없는 환경이면 `CUDA available: False`로 나올 수 있음 → 속도 느림.
- `--eval` 옵션은 현재 mmdet3d의 `.mim/tools/test.py`에서 지원하지 않음(에러 발생) → 제거 필요.

이 노트 흐름대로 실행하면, 어느 단계에서 포맷이 부족한지, GT가 없어서 평가가 안 되는지 바로 확인할 수 있습니다. GT 변환까지 필요하면 `instances` 생성용 변환 스크립트가 추가로 필요합니다.***
