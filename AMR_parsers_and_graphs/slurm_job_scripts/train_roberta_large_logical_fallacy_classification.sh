#!/bin/bash
#SBATCH --job-name=logical_fallacy_classifier
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
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
    --experiment train \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_type csv \
    --input_feature source_article

python main_classifier.py \
    --experiment train \
    --train_input_file data/edu_train.csv \
    --dev_input_file data/edu_dev.csv \
    --test_input_file data/edu_test.csv \
    --input_type csv \
    --input_feature masked_articles


# python main_classifier.py \
#     --experiment train \
#     --train_input_file tmp/masked_sentences_with_AMR_container_objects_train.joblib \
#     --dev_input_file tmp/masked_sentences_with_AMR_container_objects_dev.joblib \
#     --test_input_file tmp/masked_sentences_with_AMR_container_objects_test.joblib \
#     --input_type amr
    
# python main_classifier.py \
#     --experiment case_augmented_training \
#     --train_input_file tmp/masked_sentences_with_AMR_container_objects_train.joblib \
#     --dev_input_file tmp/masked_sentences_with_AMR_container_objects_dev.joblib \
#     --test_input_file tmp/masked_sentences_with_AMR_container_objects_test.joblib \
#     --input_type amr \
#     --gcn_model_path gcn_model.pt \
#     --num_cases 1


# python main_classifier.py \
#     --experiment train \
#     --train_input_file tmp/masked_sentences_with_AMR_container_objects_with_label2words_wordnet_conceptnet.joblib \
#     --dev_input_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words_wordnet_conceptnet.joblib \
#     --test_input_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words_wordnet_conceptnet.joblib \
#     --input_type amr \
#     --augments wordnet\&conceptnet

# python main_classifier.py \
#     --experiment train \
#     --train_input_file tmp/masked_sentences_with_AMR_container_objects_with_label2words_wordnet_conceptnet.joblib \
#     --dev_input_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words_wordnet_conceptnet.joblib \
#     --test_input_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words_wordnet_conceptnet.joblib \
#     --input_type amr \
#     --augments conceptnet

# python main_classifier.py \
#     --experiment case_augmented_training \
#     --train_input_file data/edu_train.csv \
#     --dev_input_file data/edu_dev.csv \
#     --test_input_file data/edu_test.csv \
#     --input_type csv

conda deactivate