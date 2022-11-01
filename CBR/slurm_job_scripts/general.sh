#!/bin/bash
#SBATCH --job-name=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10480
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
    --retriever_type ${retriever_type:="default"} \
    --learning_rate ${learning_rate:="default"} \
    --num_epochs ${num_epochs:="default"} \
    --sweep ${sweep:="default"} \
    --classifier_dropout ${classifier_dropout:="default"} \
    --mid_layer_dropout ${mid_layer_dropout:="default"} \
    --gcn_layers ${gcn_layers:="default"} \
    --gcn_model_path ${gcn_model_path:="default"} \
    --weight_decay ${weight_decay:="default"} \
    --augments ${augments:="default"} \
    --g_type ${g_type:="default"} \
    --cbr ${cbr:="default"} \
    --num_cases ${num_cases:="default"} \
    --all_good_cases ${all_good_cases:="default"} \
    --all_bad_cases ${all_bad_cases:="default"} \
    --cbr_threshold ${cbr_threshold:="default"} \
    --checkpoint ${checkpoint:="default"} \
    --predictions_path ${predictions_path:="default"} \
    --encoder_dropout_rate ${encoder_dropout_rate:="default"} \
    --attn_dropout_rate ${attn_dropout_rate:="default"} \
    --last_layer_dropout ${last_layer_dropout:="default"} \
    --data_dir ${data_dir:="default"}



conda deactivate