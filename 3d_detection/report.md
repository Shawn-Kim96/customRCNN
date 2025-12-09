# 3D Detection Report (Waymo vs nuScenes)

Su Hyun Kim (ID: 018219422)

## Setup
- **Env**: `conda activate det3d310` (PyTorch + mmdet3d), CUDA GPU.
- **Commands**  
  ```
  # One line execution
  sbatch scripts/run_final_experiment.sbatch

  # Inference (GT enabled for mAP)
  python unified_inference_vis.py \
    --waymo-data data_waymo_kitti \
    --waymo-config checkpoints/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-car.py \
    --waymo-checkpoint checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth \
    --waymo-samples 794 \
    --nuscenes-data data_nuscenes \
    --nuscenes-config checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    --nuscenes-checkpoint checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --nuscenes-samples 500 \
    --require-gt --output-dir final_results_15916 --fps 10 --save-ply --save-vis --create-video

  # Analysis (mAP/PR with yaw-aware IoU)
  python metrics_analysis.py \
    --results-dir final_results_15916 \
    --output-dir final_analysis_15916_fix
  ```

## Models & Datasets
- **Waymo_PointPillars**: single-class car detector, training split (794 samples), KITTI-format LiDAR + camera.
- **nuScenes_CenterPoint**: 10-class detector, val subset (500 samples), LiDAR + camera.

## Metrics (single concise table)

| Model/Dataset | mAP | Precision@0.5 | Recall@0.5 | FPS | Latency (ms) | Avg Confidence | Samples |
|:--------------|----:|--------------:|-----------:|----:|-------------:|---------------:|--------:|
| Waymo_PointPillars | 0.263 | 0.068 | 0.067 | 3.50 | 285.5 | 0.503 | 794 |
| nuScenes_CenterPoint | 0.005 | 0.003 | 0.012 | 1.77 | 566.2 | 0.237 | 500 |


## Results
- There is representative screenshots of camera view under `results/`. For full images including Lidar screenshot, the images are under `final_results_15916/waymo_pointpillars/frames/*.png` and `final_results_15916/nuscenes_centerpoint/frames/*.png`.
- The video was too large to upload in github, so it is uploaded in google drive ([Nuscenes video](https://drive.google.com/file/d/1u-91BSosz6lsG3f4l7Zyt3-m5nsYy_nA/view?usp=drive_link), [Waymo video](https://drive.google.com/file/d/1GzBtuhoNPs6r5hpR6uV6mGIHkf83af9E/view?usp=drive_link)). You can also check the video by accessing hpc to `/home/018219422/customRCNN/3d_detection/results`.

## Takeaways
- **Speed vs. breadth**: PointPillars runs ~2× faster (3.5 FPS) and with lower latency; CenterPoint is heavier due to multi-class voxel backbone and larger head.

- **Precision limits**: Both models have low Precision@0.5 (0.068 / 0.003) and low mAP, indicating many low-confidence/false positives. Because we are using the pretrained model, there might be a problem in loading data or executing the model.

- **mAP gap**: Waymo mAP 0.263 vs nuScenes 0.005. CenterPoint struggles because our yaw-aware 3D IoU at 0.5/0.7 is strict on crowded, rotated objects. Also class imbalance and small-object prevalence hurt recall.


## Limitations
- After reviewing the camera footage, I noticed that projecting all 360° LiDAR BEV detections onto a single camera view makes the frame appear cluttered. I planned to clean up the visualization code using techniques like frustum filtering, but this iteration did not incorporate those improvements. I intend to fix this in the future when time permits.
- In several frames, the 3D detection boxes have the correct size but appear slightly misaligned. This may be due to calibration issues or because the model is not fully adapted to the domain/training data.
