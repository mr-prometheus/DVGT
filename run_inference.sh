#!/bin/bash
#SBATCH -J DVGT_Inference
#SBATCH --mem=24GB  
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/dvgt_inference.out

# Manually add your conda environment to PATH
export PATH="/home/de575594/.conda/envs/dvgt/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/12.4
module load gcc/9

conda activate dvgt

# Install requirements if needed (uncomment if first run)
# pip install -r requirements.txt

# Step 1: Download DINOv3 weights if not already present (run once)
# Uncomment the following line on first run:
# python download_v3_weights.py --hf_token "" --method huggingface_hub

# Step 2: Run inference
# CUDA_LAUNCH_BLOCKING=1 python inference.py \
#     --csv_path /home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/metadata_test.csv \
#     --frames_dir /home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_frames/frames_4s \
#     --checkpoint_path models/DVGT/open_ckpt.pt \
#     --output_dir /home/de575594/Deepan/CV/geolocalization/vggt-long/outputs/dvgt_inference \
#     --num_videos 10 \
#     --max_frames 16 \
#     --start_frame 0 \
#     --dinov3_weights_path /home/de575594/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

CUDA_LAUNCH_BLOCKING=1 python inference.py