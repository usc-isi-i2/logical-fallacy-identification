#!/bin/bash
#SBATCH --job-name=nli_curriculum_learning
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
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


python -c "print('\n\nStarting with BERT training\n\n')"
python train_three_stages.py  --model_name="textattack/bert-base-uncased-MNLI" --directory_big_bench="big_bench_training/bert" --directory_coarse="coarse_training/bert" --directory_fine="fine_training/bert" --big_bench_train_dataset="curriculum_learning_training/datasets/big_bench_train.csv" --big_bench_val_dataset="curriculum_learning_training/datasets/big_bench_val.csv" --big_bench_test_dataset="curriculum_learning_training/datasets/big_bench_test.csv" --coarse_train_dataset="curriculum_learning_training/datasets/coarse_grained_train.csv" --coarse_val_dataset="curriculum_learning_training/datasets/coarse_grained_val.csv" --coarse_test_dataset="curriculum_learning_training/datasets/coarse_grained_test.csv" --fine_train_dataset="curriculum_learning_training/datasets/fine_grained_train.csv" --fine_val_dataset="curriculum_learning_training/datasets/fine_grained_val.csv" --fine_test_dataset="curriculum_learning_training/datasets/fine_grained_test.csv" --tokenizer="textattack/bert-base-uncased-MNLI"  --num_runs=3

python -c "print('\n\nStarting with Deberta training\n\n')"
python train_three_stages.py  --model_name="cross-encoder/nli-deberta-base" --directory_big_bench="big_bench_training/deberta" --directory_coarse="coarse_training/deberta"  --directory_fine="fine_training/deberta" --big_bench_train_dataset="curriculum_learning_training/datasets/big_bench_train.csv" --big_bench_val_dataset="curriculum_learning_training/datasets/big_bench_val.csv" --big_bench_test_dataset="curriculum_learning_training/datasets/big_bench_test.csv" --coarse_train_dataset="curriculum_learning_training/datasets/coarse_grained_train.csv" --coarse_val_dataset="curriculum_learning_training/datasets/coarse_grained_val.csv" --coarse_test_dataset="curriculum_learning_training/datasets/coarse_grained_test.csv" --fine_train_dataset="curriculum_learning_training/datasets/fine_grained_train.csv" --fine_val_dataset="curriculum_learning_training/datasets/fine_grained_val.csv" --fine_test_dataset="curriculum_learning_training/datasets/fine_grained_test.csv" --tokenizer="cross-encoder/nli-deberta-base" --num_runs=3

python -c "print('\n\nStarting with DistilBERT training\n\n')"
python train_three_stages.py  --model_name="typeform/distilbert-base-uncased-mnli" --directory_big_bench="big_bench_training/distilbert" --directory_coarse="coarse_training/distilbert" --directory_fine="fine_training/distilbert"  --big_bench_train_dataset="curriculum_learning_training/datasets/big_bench_train.csv" --big_bench_val_dataset="curriculum_learning_training/datasets/big_bench_val.csv" --big_bench_test_dataset="curriculum_learning_training/datasets/big_bench_test.csv" --coarse_train_dataset="curriculum_learning_training/datasets/coarse_grained_train.csv" --coarse_val_dataset="curriculum_learning_training/datasets/coarse_grained_val.csv" --coarse_test_dataset="curriculum_learning_training/datasets/coarse_grained_test.csv" --fine_train_dataset="curriculum_learning_training/datasets/fine_grained_train.csv" --fine_val_dataset="curriculum_learning_training/datasets/fine_grained_val.csv" --fine_test_dataset="curriculum_learning_training/datasets/fine_grained_test.csv" --tokenizer="typeform/distilbert-base-uncased-mnli"  --num_runs=3

python -c "print('\n\nStarting with Electra training\n\n')"
python train_three_stages.py  --model_name="howey/electra-base-mnli"  --directory_big_bench="big_bench_training/electra"  --directory_coarse="coarse_training/electra"  --directory_fine="fine_training/electra"  --big_bench_train_dataset="curriculum_learning_training/datasets/big_bench_train.csv" --big_bench_val_dataset="curriculum_learning_training/datasets/big_bench_val.csv" --big_bench_test_dataset="curriculum_learning_training/datasets/big_bench_test.csv" --coarse_train_dataset="curriculum_learning_training/datasets/coarse_grained_train.csv" --coarse_val_dataset="curriculum_learning_training/datasets/coarse_grained_val.csv" --coarse_test_dataset="curriculum_learning_training/datasets/coarse_grained_test.csv" --fine_train_dataset="curriculum_learning_training/datasets/fine_grained_train.csv" --fine_val_dataset="curriculum_learning_training/datasets/fine_grained_val.csv" --fine_test_dataset="curriculum_learning_training/datasets/fine_grained_test.csv" --tokenizer="howey/electra-base-mnli"  --num_runs=3

python -c "print('\n\nStarting with Roberta training\n\n')"
python train_three_stages.py  --model_name="cross-encoder/nli-roberta-base"  --directory_big_bench="big_bench_training/roberta"  --directory_coarse="coarse_training/roberta"  --directory_fine="fine_training/roberta"  --big_bench_train_dataset="curriculum_learning_training/datasets/big_bench_train.csv" --big_bench_val_dataset="curriculum_learning_training/datasets/big_bench_val.csv" --big_bench_test_dataset="curriculum_learning_training/datasets/big_bench_test.csv" --coarse_train_dataset="curriculum_learning_training/datasets/coarse_grained_train.csv" --coarse_val_dataset="curriculum_learning_training/datasets/coarse_grained_val.csv" --coarse_test_dataset="curriculum_learning_training/datasets/coarse_grained_test.csv" --fine_train_dataset="curriculum_learning_training/datasets/fine_grained_train.csv" --fine_val_dataset="curriculum_learning_training/datasets/fine_grained_val.csv" --fine_test_dataset="curriculum_learning_training/datasets/fine_grained_test.csv" --tokenizer="cross-encoder/nli-roberta-base"  --num_runs=3


conda deactivate