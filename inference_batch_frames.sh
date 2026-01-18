#!/bin/bash
#SBATCH -J DVGT_Inference
#SBATCH --mem=24GB  
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/dvgt_inference_frames_clips.out

# Manually add your conda environment to PATH
export PATH="/home/de575594/.conda/envs/dvgt/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/12.4
module load gcc/9

conda activate dvgt

CUDA_LAUNCH_BLOCKING=1 python inference_dataset.py