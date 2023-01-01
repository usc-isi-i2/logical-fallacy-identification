#!/bin/bash
#SBATCH --job-name=logical_fallacy_classifier
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
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


# for dataset in "data/bigbench" "data/coarsegrained" "data/new_finegrained"
# do
dataset="data/bigbench"
echo "Dataset: $dataset"

# for checkpoint in 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' 'sentence-transformers/paraphrase-MiniLM-L6-v2' 'sentence-transformers/all-MiniLM-L12-v2' 'sentence-transformers/all-MiniLM-L6-v2' 'simcse' 'empathy'
# do


# python -m cbr_analyser.reasoner.classifier_with_attention_electra \
#     --data_dir ${dataset} \
#     --checkpoint ${checkpoint}

# done

for num_cases in 11 12 13 14 15 16 17 18 19 20
do


python -m cbr_analyser.reasoner.classifier_with_attention_electra \
    --data_dir ${dataset} \
    --num_cases ${num_cases}


done

conda deactivate