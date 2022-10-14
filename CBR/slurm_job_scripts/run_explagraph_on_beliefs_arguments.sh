#!/bin/bash
#SBATCH --job-name=explagraph_graph_generation
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/CBR/cbr_analyser/case_retriever/ExplaGraphs
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate explagraph

# STEPS

# STEP 1: PREPARE THE DATASET IN THE TSV FORMAT IN dev.tsv DATA DIRECTORY OF EXPLAGRAPH

# all the datasets would be ../../../cache/explagraph/
#                                     -- dev.tsv
#                                     -- internal_nodes
#                                     -- external_nodes                                    


# for split in "train" "dev" "test"
for split in "train"
do

# You should have the dev and test files in the cache/explagraph/ directory in the tsv format and right format.

rm -rf ./data/dev.tsv ./data/test.tsv
cp "../../../cache/explagraph/$split.tsv" ./data/dev.tsv
cp "../../../cache/explagraph/$split.tsv" ./data/test.tsv


# STEP 2: RUN THE test_structured_model.sh SCRIPT WITHOUT THE do_eval_edge PARAMETER TO GET THE INTERNAL_NODES

python structured_model/run_joint_model.py \
    --model_type roberta_eg \
    --model_name_or_path ./models/sp_model \
    --task_name eg \
    --do_prediction \
    --do_lower_case \
    --data_dir ./data \
    --per_gpu_eval_batch_size 8 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir ./models/sp_model \
    --seed 42 \
    --data_cache_dir ./tmp \
    --cache_dir ./tmp \
    --evaluate_during_training

# STEP 3: copy the external nodes 

rm -rf ./data/external_nodes_dev.txt
cp "../../../cache/explagraph/external_nodes_$split.tsv" ./data/external_nodes_dev.txt


# STEP 4: COPY THE EXTRACTED INTERNAL NODES TO THE INTERNAL_NODES FILE IN THE DATA DIRECTORY

rm -rf ./data/internal_nodes_dev.txt
mv ./models/sp_model/prediction_nodes_test.lst ./data/internal_nodes_dev.txt

# STEP 5: RUN THE test_structured_model.sh SCRIPT WITH THE do_eval_edge TO GET THE EDGES GENERATED

python ./structured_model/run_joint_model.py \
    --model_type roberta_eg \
    --model_name_or_path ./models/sp_model \
    --task_name eg \
    --do_eval \
    --do_eval_edge \
    --do_lower_case \
    --data_dir ./data \
    --per_gpu_eval_batch_size 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir ./models/sp_model \
    --seed 42 \
    --data_cache_dir ./tmp \
    --cache_dir ./tmp \
    --evaluate_during_training


mv ./models/sp_model/prediction_edges_dev.lst "../../../cache/explagraph/predicted_edges_$split.lst"

done


conda deactivate