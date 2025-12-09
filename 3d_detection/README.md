# 3D Object Detection Assignment - Complete Solution

Final report including matrixs are `report.md`. This document is just for showing how to execute the code.


## Quick start
In hpc environment (018219422), simply run
```
cd /home/018219422/customRCNN/3d_detection
conda activate det3d310
sbatch scripts/run_final_experiment.sbatch
```

## Files Overview

### Main Scripts
- **`unified_inference_vis.py`** - Main inference script (Waymo + nuScenes)
- **`metrics_analysis.py`** - Metrics calculation and comparison
- **`open3d_viewer.py`** - Interactive PLY file viewer

### Data
- Waymo dataset (KITTI format): `data_waymo_kitti/`
- nuScenes dataset: `data_nuscenes/`

### Models
- PointPillars checkpoint: `checkpoints/hv_pointpillars_*.pth`
- CenterPoint checkpoint: `checkpoints/centerpoint_*.pth`

## Results

### Output Structure

```
final_results_<JOB_ID>/
├── waymo_pointpillars/
│   ├── frames/*.png          # BEV visualization images
│   ├── ply/*.ply             # Point clouds with detections
│   ├── json/*.json           # Metadata
│   └── demo_waymo.mp4        # Demo video
├── nuscenes_centerpoint/
│   └── (same structure)
└── performance_summary.json

final_analysis_<JOB_ID>/
├── comparison_table.csv       # Model comparison
├── analysis_report.txt        # Detailed analysis
└── performance_comparison.png # Performance charts

logs/
├── inference_<JOB_ID>.out    # Standard output
├── inference_<JOB_ID>.err    # Error output
└── summary_<JOB_ID>.txt      # Job summary

results/                      # Best result

reports.md                    # Final report
```
