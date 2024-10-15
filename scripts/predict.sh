#!/bin/bash
#SBATCH --job-name %name
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

module --force purge
module load anaconda/3
conda init
conda activate /home/mila/f/floresl/miniconda3/envs/al

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
python predict.py "$@"