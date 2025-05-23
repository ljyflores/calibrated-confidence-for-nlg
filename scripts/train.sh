#!/bin/bash
#SBATCH --job-name %name
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere"

module --force purge
module load anaconda/3
conda init
conda activate /home/mila/f/floresl/miniconda3/envs/al

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
python train.py "$@"