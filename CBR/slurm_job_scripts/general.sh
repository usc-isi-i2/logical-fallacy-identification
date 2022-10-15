#!/bin/bash
#SBATCH --job-name=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
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


python main.py \
    --input_file ${input_file:=default} \
    --output_file ${output_file:=default} \
    --task ${task:=default} \
    --target_file ${target_file:=default} \
    --source_feature ${source_feature:=default} \
    --source_file ${source_file:=default}


conda deactivate