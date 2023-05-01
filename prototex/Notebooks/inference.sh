#!/bin/bash
#SBATCH --job-name=prototex_explanations
#SBATCH --output=slurm_execution/%x-%j.out
#SBATCH --error=slurm_execution/%x-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/himanshu.rawlani/propaganda_detection/prototex_custom/Notebooks
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

dataset="data/finegrained_with_none"
echo "dataset: ${dataset}"
modelname="curr_lf_fine_updated_aug_with_none_nli_prototex"

python inference_and_explanations.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "curriculum-learning" --experiment "lf_fine_updated_classification_1" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "BART"

conda deactivate