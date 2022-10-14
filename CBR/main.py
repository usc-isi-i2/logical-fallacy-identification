import argparse
import os
import subprocess

from cbr_analyser.consts import *
from cbr_analyser.logging.custom_logger import get_logger

if __name__ == '__main__':
    
    logger = get_logger(logger_name=f"{__name__}.{os.path.basename(__file__)}")
    
    parser = argparse.ArgumentParser()    
    
    parser.add_argument('--data_dir', type=str, default='data/')
    
    args = parser.parse_args()
    
    # Check all the arguments to be correct in the consts.py file because they are used in the whole project
    
    # first step is to get the AMR graphs from the data_dir
    if os.path.exists(PATH_TO_MASKED_SENTENCES_AMRS_TRAIN) and \
        os.path.exists(PATH_TO_MASKED_SENTENCES_AMRS_DEV) and \
            os.path.exists(PATH_TO_MASKED_SENTENCES_AMRS_TEST):
        print("AMR graphs already exist, skipping step 1")
        logger.info("AMR graphs already exist, skipping step 1")
    else:
        task = "amr_generation"
        job_id = subprocess.check_output(['sbatch', 'slurm_job_scripts/generate_amrs.sh']).decode()
        logger.info(f"Submitted job {job_id} for task {task}")
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    