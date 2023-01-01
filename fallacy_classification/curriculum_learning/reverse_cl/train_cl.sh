#!/bin/bash
#SBATCH --job-name=eval_loss_nli_reverse_curriculum_learning
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=reverse_curriculum_learning_training

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
python train_three_stages.py --model_name="textattack/bert-base-uncased-MNLI" --directory_big_bench="reverse_curriculum_learning_training/big_bench_training/bert" --directory_coarse="reverse_curriculum_learning_training/coarse_training/bert" --directory_fine="reverse_curriculum_learning_training/fine_training/bert" --tokenizer="textattack/bert-base-uncased-MNLI" --num_runs=3

python -c "print('\n\nStarting with Deberta training\n\n')"
python train_three_stages.py --model_name="cross-encoder/nli-deberta-base" --directory_big_bench="reverse_curriculum_learning_training/big_bench_training/deberta" --directory_coarse="reverse_curriculum_learning_training/coarse_training/deberta" --directory_fine="reverse_curriculum_learning_training/fine_training/deberta" --tokenizer="cross-encoder/nli-deberta-base" --num_runs=3

python -c "print('\n\nStarting with DistilBERT training\n\n')"
python train_three_stages.py --model_name="typeform/distilbert-base-uncased-mnli" --directory_big_bench="reverse_curriculum_learning_training/big_bench_training/distilbert" --directory_coarse="reverse_curriculum_learning_training/coarse_training/distilbert" --directory_fine="reverse_curriculum_learning_training/fine_training/distilbert" --tokenizer="typeform/distilbert-base-uncased-mnli" --num_runs=3

python -c "print('\n\nStarting with Electra training\n\n')"
python train_three_stages.py --model_name="howey/electra-base-mnli" --directory_big_bench="reverse_curriculum_learning_training/big_bench_training/electra" --directory_coarse="reverse_curriculum_learning_training/coarse_training/electra" --directory_fine="reverse_curriculum_learning_training/fine_training/electra" --tokenizer="howey/electra-base-mnli" --num_runs=3

python -c "print('\n\nStarting with Roberta training\n\n')"
python train_three_stages.py --model_name="cross-encoder/nli-roberta-base" --directory_big_bench="reverse_curriculum_learning_training/big_bench_training/roberta" --directory_coarse="reverse_curriculum_learning_training/coarse_training/roberta" --directory_fine="reverse_curriculum_learning_training/fine_training/roberta" --tokenizer="cross-encoder/nli-roberta-base" --num_runs=3

conda deactivate