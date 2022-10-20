#!/bin/bash
#SBATCH --job-name=sentence_segmentation
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
conda activate ds

for split in "train" "dev" "test"
do


python cbr_analyser/case_retriever/deep_segment/deepsegment_handler.py \
    --input_path "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
    --output_path "cache/masked_sentences_with_AMR_container_objects_${split}_with_segments.joblib"

done



conda deactivate