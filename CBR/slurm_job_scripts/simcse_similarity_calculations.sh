#!/bin/bash
#SBATCH --job-name=simcse_similarity_calculations
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

for split in "train" "dev" "test"
do

source_feature="masked_articles"
python -m cbr_analyser.case_retriver.transformers.simcse_similarity_calculations \
    --source_feature ${source_feature} \
    --source_file "cache/masked_sentences_with_AMR_container_objects_train.joblib" \
    --target_file "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
    --output_file "cache/simcse_similarities_${source_feature}_${split}.joblib"

source_feature="source_article"
python -m cbr_analyser.case_retriver.transformers.simcse_similarity_calculations \
    --source_feature ${source_feature} \
    --source_file "cache/masked_sentences_with_AMR_container_objects_train.joblib" \
    --target_file "cache/masked_sentences_with_AMR_container_objects_${split}.joblib" \
    --output_file "cache/simcse_similarities_${source_feature}_${split}.joblib"

done
conda deactivate