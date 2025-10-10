#!/bin/bash
#SBATCH -p gpuql   
#SBATCH -J customRCNN_vit_backbone_train    
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o %x-%j.out    
#SBATCH -e %x-%j.err
#SBATCH --time=24:00:00    

export CUDA_VISIBLE_DEVICES=0,1

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
    wait $!
    echo "Finished training for backbone: ${backbone}"
}

// run_training "vit_b_16" \
//     --batch-size 2 \
//     --pretrained-path data/pretrained_models/vit/vit_b_16_weights.pth

run_training "resnet101" \
    --batch-size 2 \
    --trainable-layers 2

run_training "swin_t" \
    --batch-size 2
