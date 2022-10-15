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


def follow_the_usual_process(source_feature: str):
    # first step is to get the AMR graphs from the data_dir
    for split in ["train", "dev", "test"]:
        path = f"cache/masked_sentences_with_AMR_container_objects_{split}.joblib"
        if os.path.exists(path):
            print(f"AMR graphs for the {split} split already exist, skipping step 1")
            logger.info(f"AMR graphs for the {split} split already exist, skipping step 1")
        else:
            task = "amr_generation"    
            job_id = subprocess.check_output([
                'sbatch',
                f"--export=ALL,input_file=data/edu_{split}.csv,output_file={path},task={task}",
                'slurm_job_scripts/general.sh'
            ]).decode()
            print(f"Submitted job {job_id} for {split} split")
            logger.info(
                f"Submitted job {job_id} for task {task} for split {split}")
            

    # second step is to train or prepare the case retrievers
    # First would be transformers retriever (both the simcse and sentence transformers)
    
    # There are two kinds of transformer retrievers we can use: simcse and sentence transformers
    # Also there are two ways we can do the training: use the masked sentences or use the original sentences
    
    for split in ["train", "dev", "test"]:
        path = f"cache/simcse_similarities_{source_feature}_{split}.joblib"
        if os.path.exists(path):
            print(f"simcse similarities for the {split} split already exist, skipping step 2")
            logger.info(f"simcse similarities for the {split} split already exist, skipping step 2")
        else:
            task = "simcse_similarity"
            job_id = subprocess.check_output([
                'sbatch',
                f"--export=ALL,source_feature={source_feature},source_file=cache/masked_sentences_with_AMR_container_objects_train.joblib,target_file=cache/masked_sentences_with_AMR_container_objects_{split}.joblib,output_file={path},task={task}",
                'slurm_job_scripts/general.sh'
            ]).decode()
            print(f"Submitted job {job_id} for {split} split")
            logger.info(
                f"Submitted job {job_id} for task {task} for split {split}")
            
            
            
    
    

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--source_feature', choices=['masked', 'source'], type=str, default='masked')
    # TODO: add all the choices for the tasks
    parser.add_argument('--task', type=str)
    
    # Check all the arguments to be correct in the consts.py file because they are used in the whole project
    args = parser.parse_args()
    
    
    if args.task is None:
        follow_the_usual_process(source_feature=args.source_feature)

    
    
    
    
    