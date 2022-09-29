#!/bin/bash
#SBATCH --job-name=logical_fallacy_classifier
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:v100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/AMR_parsers_and_graphs
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


# TODO: Should be revised based on the names of the files \
#  augmenting wordnet and conceptNet to the main datastes


python main_classifier.py \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_type csv \
    --input_feature source_article \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --classifier_dropout 0.1 \
    --weight_decay 0.001

python main_classifier.py \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_type csv \
    --input_feature masked_articles \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --classifier_dropout 0.1 \
    --weight_decay 0.01



conda deactivate