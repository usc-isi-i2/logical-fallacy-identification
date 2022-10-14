#!/bin/bash
#SBATCH --job-name=amr_graph_augmentation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/CBR
# Verify working directory
echo $(pwd)

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate general

# WORDNET

# for split in "train" "dev" "test"
# do

# python wordnet_augmentation.py \
#     --input_file "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
#     --output_file "cache/masked_sentences_with_AMR_container_objects_${split}_wordnet.joblib" \

# done

# # CONCEPTNET

for split in "train" "dev" "test"
do

python -m cbr_analyser.augmentations.conceptnet_augmentation \
    --input_file "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
    --output_file "cache/masked_sentences_with_AMR_container_objects_${split}_conceptnet_good_relations.joblib" \
    --rel_file data/conceptNet_relations.joblib \
    --label_file data/conceptNet_labels.joblib \
    --good_relations

python -m cbr_analyser.augmentations.conceptnet_augmentation \
    --input_file "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
    --output_file "cache/masked_sentences_with_AMR_container_objects_${split}_conceptnet.joblib" \
    --rel_file data/conceptNet_relations.joblib \
    --label_file data/conceptNet_labels.joblib

done


conda deactivate