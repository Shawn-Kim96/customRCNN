#!/bin/bash
#SBATCH -p gpuql   
#SBATCH -J customRCNN_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o %x-%j.out    
#SBATCH -e %x-%j.err
#SBATCH --time=24:00:00    

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Allocated GPUs:"
nvidia-smi

timestamp=$(date +"%y%m%dT%H%M%S")
mkdir -p logs

run_training () {
    local backbone="$1"
    shift
    local logfile="logs/${timestamp}_bbt-${backbone}_train.log"
    echo "Starting training for backbone: ${backbone}"
    echo "Logging to: ${logfile}"

    nohup python src/main.py \
        --backbone "$backbone" \
        "$@" \
        > "$logfile" 2>&1 &

    TRAIN_PID=$!
    echo "Training started (PID: $TRAIN_PID)"
    
    sleep 300
    echo "------ GPU usage after 300 seconds ------"
    nvidia-smi

    wait $TRAIN_PID
    echo "Finished training for backbone: ${backbone}"
}

# run_training "resnet101" \
#     --batch-size 4 \
#     --trainable-layers 2 \
#     --resume "/home/018219422/customRCNN/src/result/standard_backbone_resnet101/ep20_lr0.005_mom0.9_wd0.0005/final_model.pth"

# run_training "vit_b_16" \
#     --batch-size 4 \
#     --epochs 20 \
#     --resume  "/home/018219422/customRCNN/src/result/advanced_backbone_vit_b_16/ep20_lr0.005_mom0.9_wd0.0005/final_model.pth"


run_training "swin_t" \
    --batch-size 4 \
    --epochs 20 \
    --resume "/home/018219422/customRCNN/src/result/advanced_backbone_swin_t/ep20_lr0.005_mom0.9_wd0.0005/final_model.pth"
