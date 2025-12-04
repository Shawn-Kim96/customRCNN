#!/bin/bash
#SBATCH -J waymo_pp_val
#SBATCH -p gpuqm
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /home/018219422/customRCNN/3d_detection/logs/waymo_pp_val_%j.out
#SBATCH -e /home/018219422/customRCNN/3d_detection/logs/waymo_pp_val_%j.err

set -euo pipefail

PYTHON="$HOME/miniconda3/envs/det3d310/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON. Adjust path to your env." >&2
  exit 1
fi

REPO_ROOT=/home/018219422/customRCNN
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/logs"

WAYMO_ROOT=/home/018219422/customRCNN/3d_detection/data_waymo_kitti/kitti_format
WAYMO_PARENT="$(dirname "$WAYMO_ROOT")"
WAYMO_BIN="$WAYMO_PARENT/waymo_format/gt.bin"
CFG=/home/018219422/customRCNN/3d_detection/checkpoints/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-car.py

CKPT=/home/018219422/customRCNN/3d_detection/checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth
WORKDIR=/home/018219422/customRCNN/3d_detection/results/waymo_pointpillars_val

mkdir -p "$WORKDIR"

if [ ! -f "$WAYMO_BIN" ]; then
  echo "Waymo GT binary not found at $WAYMO_BIN (create waymo_format/gt.bin alongside kitti_format)." >&2
  exit 1
fi

echo "PWD: $(pwd)"
echo "Using CFG: $CFG"
ls -l "$CFG"

set -x
"$PYTHON" -m mim test mmdet3d \
  "$CFG" \
  --checkpoint "$CKPT" \
  --work-dir "$WORKDIR" \
  --cfg-options \
    test_dataloader.dataset.data_root="$WAYMO_ROOT" \
    test_dataloader.dataset.ann_file="$WAYMO_ROOT/waymo_infos_val.pkl" \
    test_dataloader.dataset.data_prefix.pts=training/velodyne \
    test_evaluator.waymo_bin_file="$WAYMO_BIN" \
    val_dataloader.dataset.data_root="$WAYMO_ROOT" \
    val_dataloader.dataset.ann_file="$WAYMO_ROOT/waymo_infos_val.pkl" \
    val_dataloader.dataset.data_prefix.pts=training/velodyne \
    val_evaluator.waymo_bin_file="$WAYMO_BIN" \
    train_dataloader.dataset.dataset.data_root="$WAYMO_ROOT" \
    train_dataloader.dataset.dataset.data_prefix.pts=training/velodyne \
    test_dataloader.batch_size=1 \
    test_dataloader.num_workers=2
set +x
