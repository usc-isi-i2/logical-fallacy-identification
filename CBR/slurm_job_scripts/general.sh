#!/bin/bash
#SBATCH --job-name=general
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

python main.py \
    --input_file ${input_file:="default"} \
    --output_file ${output_file:="default"} \
    --task ${task:="default"} \
    --target_file ${target_file:="default"} \
    --source_feature ${source_feature:="default"} \
    --source_file ${source_file:="default"} \
    --train_input_file ${train_input_file:="default"} \
    --dev_input_file ${dev_input_file:="default"} \
    --test_input_file ${test_input_file:="default"} \
    --batch_size ${batch_size:="default"} \
    --learning_rate ${learning_rate:="default"} \
    --num_epochs ${num_epochs:="default"} \
    --classifier_dropout ${classifier_dropout:="default"} \
    --mid_layer_dropout ${mid_layer_dropout:="default"} \
    --gcn_layers ${gcn_layers:="default"} \
    --gcn_model_path ${gcn_model_path:="default"} \
    --weight_decay ${weight_decay:="default"} \
    --augments ${augments:="default"} \
    --g_type ${g_type:="default"} \
    --cbr ${cbr:="default"} \
    --similarity_matrices_path_train ${similarity_matrices_path_train:="default"} \
    --similarity_matrices_path_dev ${similarity_matrices_path_dev:="default"} \
    --similarity_matrices_path_test ${similarity_matrices_path_test:="default"} \
    --num_cases ${num_cases:="default"} \
    --all_good_cases ${all_good_cases:="default"} \
    --all_bad_cases ${all_bad_cases:="default"} \
    --cbr_threshold ${cbr_threshold:="default"} \
    --checkpoint ${checkpoint:="default"} \
    --predictions_path ${predictions_path:="default"}



conda deactivate