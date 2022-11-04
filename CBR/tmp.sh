#!/bin/bash
#SBATCH --job-name=logical_fallacy_classifier
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/CBR
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate general

dataset="data/finegrained"
echo "Dataset: $dataset"

python tmp.py \
    --data_dir ${dataset}

conda deactivate