#!/bin/bash
#SBATCH --job-name=logical_fallacy_classifier
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

python -m cbr_analyser.reasoner.baseline_classifier \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_feature masked_articles


python -m cbr_analyser.reasoner.baseline_classifier \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_feature source_article


conda deactivate