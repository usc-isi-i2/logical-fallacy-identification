#!/bin/bash
#SBATCH --job-name=gpt_j_prompts_extraction
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
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


python -m cbr_analyser.augmentations.prompt_gpt_j \
    --task generate \
    --input_file cache/masked_sentences_with_AMR_container_objects.joblib \
    --output_file cache/masked_sentences_with_AMR_container_objects_with_belief_argument.joblib


python -m cbr_analyser.augmentations.prompt_gpt_j \
    --task generate \
    --input_file cache/masked_sentences_with_AMR_container_objects_dev.joblib \
    --output_file cache/masked_sentences_with_AMR_container_objects_dev_with_belief_argument.joblib


python -m cbr_analyser.augmentations.prompt_gpt_j \
    --task generate \
    --input_file cache/masked_sentences_with_AMR_container_objects_test.joblib \
    --output_file cache/masked_sentences_with_AMR_container_objects_test_with_belief_argument.joblib



conda deactivate