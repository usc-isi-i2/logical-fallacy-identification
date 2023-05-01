#!/bin/bash
#SBATCH --job-name=curr_nli_electra_prototex
#SBATCH --output=slurm_execution/%x-%j.out
#SBATCH --error=slurm_execution/%x-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/himanshu.rawlani/propaganda_detection/prototex_custom
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate prototex

echo "Starting training with parameters:"

dataset="data/finegrained_with_none"
echo "dataset: ${dataset}"
modelname="curr_finegrained_nli_electra_prototex"
echo "modelname: ${modelname}"
model_checkpoint="Models/curr_coarsegrained_nli_electra_prototex"

for trial in 2 3
do
echo "Trail number: ${trial}"
python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "curriculum-learning" --experiment "electra_finegrained_classification_$((trial))" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra" --model_checkpoint ${model_checkpoint}
done

# for num_prototypes in 5 10 15 30 50 70 100 150
# do
# echo "number of prototypes: ${num_prototypes}"
# num_neg_prototypes=num_prototypes/10
# echo "number of positive prototypes: $((num_prototypes-num_neg_prototypes))"
# python main.py --num_prototypes ${num_prototypes} --num_pos_prototypes $((num_prototypes-num_neg_prototypes)) --data_dir ${dataset} --modelname ${modelname} --project "direct-fine-tuning" --experiment "prototypes_electra_finegrained_classification_$((num_prototypes))" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra"
# done

# srun --job-name=nli_roberta_prototex --partition=nodes --time=3-00:00:00 --cpus-per-task=8 --mem=10240 --gres=gpu:a100:1 --chdir=/cluster/raid/home/himanshu.rawlani/propaganda_detection/prototex_custom --pty /bin/bash

# dataset="data/coarsegrained_with_none"
# echo "dataset: ${dataset}"
# modelname="coarsegrained_nli_electra_prototex"
# echo "modelname: ${modelname}"
# # model_checkpoint="Models/finegrained_nli_electra_prototex"

# for trial in 2 3
# do
# echo "Trail number: ${trial}"
# python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "direct-fine-tuning" --experiment "electra_coarsegrained_classification_$((trial))" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra"
# # --model_checkpoint ${model_checkpoint}
# done

# dataset="data/bigbench"
# echo "dataset: ${dataset}"
# modelname="binary_nli_electra_prototex"
# echo "modelname: ${modelname}"
# # model_checkpoint="Models/rev_curr_coarsegrained_nli_electra_prototex"

# for trial in 2 3
# do
# echo "Trail number: ${trial}"
# python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "direct-fine-tuning" --experiment "electra_binary_classification_$((trial))" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra"
# # --model_checkpoint ${model_checkpoint}
# done

# python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir "data/finegrained_with_none" --modelname "finegrained_nli_electra_prototex" --project "direct-fine-tuning" --experiment "electra_finegrained_classification_1" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "No" --architecture "Electra"

conda deactivate
