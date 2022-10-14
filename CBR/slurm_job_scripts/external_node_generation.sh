#!/bin/bash
#SBATCH --job-name=external_node_generation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:v100:1
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

# Train

python prompt_gpt_j.py \
    --task external_node_generation \
    --input_file tmp/masked_sentences_with_AMR_container_objects_with_belief_argument.joblib \
    --output_file tmp/explagraph/external_nodes_train.tsv

# Dev

# python prompt_gpt_j.py \
#     --task external_node_generation \
#     --input_file tmp/masked_sentences_with_AMR_container_objects_dev_with_belief_argument.joblib \
#     --output_file tmp/explagraph/external_nodes_dev.tsv

# # Test

# python prompt_gpt_j.py \
#     --task external_node_generation \
#     --input_file tmp/masked_sentences_with_AMR_container_objects_test_with_belief_argument.joblib \
#     --output_file tmp/explagraph/external_nodes_test.tsv



conda deactivate