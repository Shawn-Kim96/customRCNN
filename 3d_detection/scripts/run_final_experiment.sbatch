#!/bin/bash
#SBATCH --job-name=3d_det_final
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpuqm

set -e  # Exit on error

echo "================================================================================"
echo "3D OBJECT DETECTION - FINAL EXPERIMENT"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Waymo: 794 samples (training, with GT for mAP)"
echo "nuScenes: 500 samples (with GT)"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Create logs directory
mkdir -p logs

# Activate conda environment
echo "[1/5] Activating conda environment: det3d310"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det3d310

# Check environment
echo ""
echo "[2/5] Environment check..."
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
fi
echo ""

# Count samples
WAYMO_TRAIN_COUNT=$(find data_waymo_kitti/kitti_format/training/velodyne/ -name "*.bin" 2>/dev/null | wc -l || echo 0)
echo "[3/5] Dataset information:"
echo "  Waymo training samples: ${WAYMO_TRAIN_COUNT} (processing first 794 for mAP)"
echo "  nuScenes samples: 500 (out of ~36,737, with GT)"
echo ""

# Set output directories
OUTPUT_DIR="final_results_${SLURM_JOB_ID}"
ANALYSIS_DIR="final_analysis_${SLURM_JOB_ID}"

echo "Output directories:"
echo "  Results: ${OUTPUT_DIR}"
echo "  Analysis: ${ANALYSIS_DIR}"
echo ""

# ============================================================================
# Run inference on both datasets
# ============================================================================
echo "[4/5] Running inference on both datasets..."
echo "  Waymo: 794 samples (training split for GT)"
echo "  nuScenes: 500 samples (train/val with GT)"
echo ""

python unified_inference_vis.py \
  --waymo-data data_waymo_kitti \
  --waymo-config checkpoints/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-car.py \
  --waymo-checkpoint checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth \
  --nuscenes-data data_nuscenes \
  --nuscenes-config checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
  --nuscenes-checkpoint checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
  --output-dir ${OUTPUT_DIR} \
  --waymo-samples 794 \
  --nuscenes-samples 500 \
  --device cuda:0 \
  --require-gt \
  --save-ply \
  --save-vis \
  --create-video \
  --fps 10

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo ""
echo "Inference completed successfully!"
echo ""

# ============================================================================
# Generate analysis
# ============================================================================
echo "[5/5] Generating comparative analysis..."
python metrics_analysis.py \
  --results-dir ${OUTPUT_DIR} \
  --output-dir ${ANALYSIS_DIR}

# Count outputs
WAYMO_FRAMES=$(find ${OUTPUT_DIR}/waymo_pointpillars/frames/ -name "*.png" 2>/dev/null | wc -l || echo 0)
WAYMO_PLY=$(find ${OUTPUT_DIR}/waymo_pointpillars/ply/ -name "*.ply" 2>/dev/null | wc -l || echo 0)
NUSCENES_FRAMES=$(find ${OUTPUT_DIR}/nuscenes_centerpoint/frames/ -name "*.png" 2>/dev/null | wc -l || echo 0)
NUSCENES_PLY=$(find ${OUTPUT_DIR}/nuscenes_centerpoint/ply/ -name "*.ply" 2>/dev/null | wc -l || echo 0)

echo ""
echo "================================================================================"
echo "FINAL EXPERIMENT COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "End time: $(date)"
echo ""
echo "RESULTS SUMMARY:"
echo "----------------"
echo "Waymo (PointPillars):"
echo "  - Samples processed: ${WAYMO_FRAMES}"
echo "  - PLY files: ${WAYMO_PLY}"
echo "  - Video: ${OUTPUT_DIR}/waymo_pointpillars/demo_waymo.mp4"
echo ""
echo "nuScenes (CenterPoint):"
echo "  - Samples processed: ${NUSCENES_FRAMES}"
echo "  - PLY files: ${NUSCENES_PLY}"
echo "  - Video: ${OUTPUT_DIR}/nuscenes_centerpoint/demo_nuscenes.mp4"
echo ""
echo "Analysis:"
echo "  - Report (Markdown): ${ANALYSIS_DIR}/analysis_report.md"
echo "  - Table (CSV): ${ANALYSIS_DIR}/comparison_table.csv"
echo "  - Plot (PNG): ${ANALYSIS_DIR}/performance_comparison.png"
echo ""
echo "View the report with:"
echo "  cat ${ANALYSIS_DIR}/analysis_report.md"
echo "  # Or open in a markdown viewer"
echo ""
echo "================================================================================"
