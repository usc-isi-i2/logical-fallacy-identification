#!/bin/bash
#SBATCH --job-name=train_predict_GCN
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

# python -m cbr_analyser.case_retriever.gcn.gcn \
#     --task train \
#     --train_input_file cache/masked_sentences_with_AMR_container_objects_train.joblib \
#     --dev_input_file cache/masked_sentences_with_AMR_container_objects_dev.joblib \
#     --test_input_file cache/masked_sentences_with_AMR_container_objects_test.joblib \
#     --model_path gcn_model.pt

python -m cbr_analyser.case_retriever.gcn.gcn \
    --task predict \
    --train_input_file cache/masked_sentences_with_AMR_container_objects_train.joblib \
    --dev_input_file cache/masked_sentences_with_AMR_container_objects_dev.joblib \
    --test_input_file cache/masked_sentences_with_AMR_container_objects_test.joblib \
    --predictions_path gcn_results \
    --model_path gcn_model.pt


# python -m cbr_analyser.case_retriever.gcn.gcn \
#     --task hptuning \
#     --all_data cache/masked_sentences_with_AMR_container_objects_all.joblib \
#     --train_input_file cache/masked_sentences_with_AMR_container_objects_train.joblib \
#     --dev_input_file cache/masked_sentences_with_AMR_container_objects_dev.joblib \
#     --test_input_file cache/masked_sentences_with_AMR_container_objects_test.joblib \
#     --model_path gcn_model.pt


conda deactivate