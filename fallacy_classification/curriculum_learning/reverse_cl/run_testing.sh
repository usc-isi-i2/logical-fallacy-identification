#!/bin/bash
#SBATCH --job-name=curriculum_learning
#SBATCH --output=testing-%x-%j.out
#SBATCH --error=testing-%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=curriculum_learning_training

# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
conda activate virtual-env


# -----------------------------------
python -c "print('\n\nStarting with Bert base Evaluation with 16 classes\n\n')"
python test_metrics.py --model_path="curriculum_learning_training/fine_training/bert/checkpoint-1057" --tokenizer="sentence-transformers/nli-bert-base"

python -c "print('\n\nStarting with Deberta base Evaluation with 16 classes\n\n')"
python test_metrics.py --model_path="curriculum_learning_training/fine_training/deberta/checkpoint-1812" --tokenizer="cross-encoder/nli-deberta-base"

python -c "print('\n\nStarting with Distilbert base Evaluation with 16 classes\n\n')"
python test_metrics.py --model_path="curriculum_learning_training/fine_training/distilbert/checkpoint-906" --tokenizer="sentence-transformers/nli-distilbert-base"

python -c "print('\n\nStarting with Roberta base Evaluation with 16 classes\n\n')"
python test_metrics.py --model_path="curriculum_learning_training/fine_training/roberta/checkpoint-1208" --tokenizer="cross-encoder/nli-roberta-base"

conda deactivate