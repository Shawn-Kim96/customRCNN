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
backboneName="vit_b_16"       
logfile="logs/${timestamp}_bbt-${backboneName}_train.log"

nohup python src/main.py \
    --backbone "$backboneName" \
    --batch-size 2\
    --pretrained-path data/pretrained_models/vit/vit_b_16_weights.pth \
    > "$logfile" 2>&1 &
wait
