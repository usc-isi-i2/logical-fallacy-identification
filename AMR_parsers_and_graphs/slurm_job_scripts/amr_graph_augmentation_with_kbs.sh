#!/bin/bash
#SBATCH --job-name=amr_graph_augmentation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/AMR_parsers_and_graphs
# Verify working directory
echo $(pwd)

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate general

# WORDNET
# Dev split
# python wordnet_augmentation.py \
#     --input_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words.joblib \
#     --output_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words_wordnet.joblib

# # Test split
# python wordnet_augmentation.py \
#     --input_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words.joblib \
#     --output_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words_wordnet.joblib


# CONCEPTNET
# Dev split
python conceptnet_augmentation.py \
    --input_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words_wordnet.joblib \
    --output_file tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words_wordnet_conceptnet.joblib

# Test split
python conceptnet_augmentation.py \
    --input_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words_wordnet.joblib \
    --output_file tmp/masked_sentences_with_AMR_container_objects_test_with_label2words_wordnet_conceptnet.joblib


conda deactivate