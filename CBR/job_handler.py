import argparse
import os
import subprocess
import sys

import joblib
from IPython import embed

from cbr_analyser.consts import *
from cbr_analyser.logging.custom_logger import get_logger

logger = get_logger(logger_name=f"{__name__}.{os.path.basename(__file__)}")

import os
import sys


def do_amr_generation(args, split, path): 
    envs = "--export=ALL,"
    envs += f"input_file=data/edu_{split}.csv,"
    envs += f"output_file={path},"
    envs += f"task={args.task}"
    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for {split} split")
    logger.info(
        f"Submitted job {job_id} for task {args.task} for split {split}")


def do_simcse_similarity(args, split, path):
    envs = "--export=ALL,"
    envs += f"source_feature={args.source_feature},"
    envs += f"source_file=cache/masked_sentences_with_AMR_container_objects_train.joblib,"
    envs += f"target_file=cache/masked_sentences_with_AMR_container_objects_{split}.joblib,"
    envs += f"output_file={path},"
    envs += f"task={args.task}"
    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for {split} split")
    logger.info(
        f"Submitted job {job_id} for task {args.task} for split {split}")


def do_train_gcn(args, split, path):
    raise NotImplementedError()


def do_train_main_classifier(args, path):
    envs = "--export=ALL,"
    envs += f"source_feature={args.source_feature},"
    envs += f"task={args.task},"
    envs += "train_input_file=cache/masked_sentences_with_AMR_container_objects_train.joblib,"
    envs += "dev_input_file=cache/masked_sentences_with_AMR_container_objects_dev.joblib,"
    envs += "test_input_file=cache/masked_sentences_with_AMR_container_objects_test.joblib,"
    envs += "batch_size=8,"
    envs += "learning_rate=2e-5,"
    envs += "num_epochs=5,"
    envs += f"cbr={args.cbr},"
    envs += "num_cases=1,"
    envs += f"similarity_matrices_path_train=cache/simcse_similarities_{args.source_feature}_train.joblib,"
    envs += f"similarity_matrices_path_dev=cache/simcse_similarities_{args.source_feature}_dev.joblib,"
    envs += f"similarity_matrices_path_test=cache/simcse_similarities_{args.source_feature}_test.joblib,"
    envs += "classifier_dropout=0.1,"
    envs += "weight_decay=0.01,"
    envs += f"predictions_path={path}"

    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for task {args.task} with args {args}")
    logger.info(
        f"Submitted job {job_id} for task {args.task} with args {args}")

def follow_the_usual_process(args):
    # first step is to get the AMR graphs from the data_dir
    for split in ["train", "dev", "test"]:
        path = f"cache/masked_sentences_with_AMR_container_objects_{split}.joblib"
        if os.path.exists(path):
            print(f"AMR graphs for the {split} split already exist, skipping step 1")
            logger.info(f"AMR graphs for the {split} split already exist, skipping step 1")
        else:
            do_amr_generation(args, split, path)
            

    # second step is to train or prepare the case retrievers
    # First would be transformers retriever (both the simcse and sentence transformers)
    
    # There are two kinds of transformer retrievers we can use: simcse and sentence transformers
    # Also there are two ways we can do the training: use the masked sentences or use the original sentences
    
    for split in ["train", "dev", "test"]:
        path = f"cache/simcse_similarities_{args.source_feature}_{split}.joblib"
        if os.path.exists(path):
            print(f"simcse similarities for the {split} split already exist, skipping step 2")
            logger.info(f"simcse similarities for the {split} split already exist, skipping step 2")
        else:
            do_simcse_similarity(args, split, path)
            
            
    # Third step is to train the case base reasoner
    do_train_main_classifier(args, args.predictions_path)


    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    

    parser.add_argument('--source_feature', choices=['masked_articles', 'source_article'], type=str, default='masked_articles')
    
    # TODO: add all the choices for the tasks
    parser.add_argument('--task', type=str, choices=['amr_generation', 'simcse_similarity', 'train_main_classifier'], default='follow_the_usual_process')
    
    parser.add_argument('--cbr', type = str, default = "False")
    parser.add_argument('--predictions_path', type = str)
    
    # Check all the arguments to be correct in the consts.py file because they are used in the whole project
    args = parser.parse_args()
    
    
    if args.task is "follow_the_usual_process":
        follow_the_usual_process(args)
        
    

    
    
    
    
    