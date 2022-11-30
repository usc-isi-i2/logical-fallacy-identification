#!/bin/bash
#SBATCH --job-name=logical_baselines
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=direct_fine_tuning

# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
conda activate virtual-env

python -c "print('\n\nStarting with BERT training\n\n')"
python train_fine_tune.py --model_name="textattack/bert-base-uncased-MNLI" --directory_big_bench="direct_fine_tuning/big_bench_training/base-bert" --directory_coarse="direct_fine_tuning/coarse_training/base-bert" --directory_fine="direct_fine_tuning/fine_training/base-bert" --tokenizer="textattack/bert-base-uncased-MNLI"

python -c "print('\n\nStarting with Deberta training\n\n')"
python train_fine_tune.py --model_name="cross-encoder/nli-deberta-base" --directory_big_bench="direct_fine_tuning/big_bench_training/base-deberta" --directory_coarse="direct_fine_tuning/coarse_training/base-deberta" --directory_fine="direct_fine_tuning/fine_training/base-deberta" --tokenizer="cross-encoder/nli-deberta-base"

python -c "print('\n\nStarting with DistilBERT training\n\n')"
python train_fine_tune.py --model_name="typeform/distilbert-base-uncased-mnli" --directory_big_bench="direct_fine_tuning/big_bench_training/base-distilbert" --directory_coarse="direct_fine_tuning/coarse_training/base-distilbert" --directory_fine="direct_fine_tuning/fine_training/base-distilbert" --tokenizer="typeform/distilbert-base-uncased-mnli"

python -c "print('\n\nStarting with Electra training\n\n')"
python train_fine_tune.py --model_name="howey/electra-base-mnli" --directory_big_bench="direct_fine_tuning/big_bench_training/electra" --directory_coarse="direct_fine_tuning/coarse_training/electra" --directory_fine="direct_fine_tuning/fine_training/electra" --tokenizer="howey/electra-base-mnli" --num_runs=2

python -c "print('\n\nStarting with Roberta training\n\n')"
python train_fine_tune.py --model_name="cross-encoder/nli-roberta-base" --directory_big_bench="direct_fine_tuning/big_bench_training/base-roberta" --directory_coarse="direct_fine_tuning/coarse_training/base-roberta" --directory_fine="direct_fine_tuning/fine_training/base-roberta" --tokenizer="cross-encoder/nli-roberta-base"

conda deactivate