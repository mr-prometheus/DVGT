#!/bin/bash
#SBATCH -J llama_download
#SBATCH --mem=12GB  
#SBATCH --output=download_job.out

module load anaconda3
module load cuda/12.4

conda activate dvgt

pip install huggingface_hub --user
python3 scripts/download_model.py