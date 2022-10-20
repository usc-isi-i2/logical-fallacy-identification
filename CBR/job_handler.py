import argparse
import os
import subprocess
import sys

import joblib
from IPython import embed

import main
from cbr_analyser.consts import *
from cbr_analyser.custom_logging.custom_logger import get_logger

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "cbr_analyser/amr/"))


logger = get_logger(logger_name=f"{__name__}.{os.path.basename(__file__)}")


def do_load_gcn(args):
    config = {
        "task": args.task
    }
    envs = "--export=ALL,"
    for key, value in config.items():
        envs += f"{key}={value},"
    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for task {args.task} with args {args}")
    logger.info(
        f"Submitted job {job_id} for task {args.task}")


def do_amr_generation(args, split, path):
    config = {
        "task": args.task,
        "output_file": path,
        "input_file": f"data/edu_{split}.csv",
    }
    envs = "--export=ALL,"
    for key, value in config.items():
        envs += f"{key}={value},"
    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for {split} split")
    logger.info(
        f"Submitted job {job_id} for task {args.task} for split {split}")


def do_empathy_similarity(args, split, path):
    config = {
        "task": args.task,
        "source_feature": args.source_feature,
        "source_file": "cache/masked_sentences_with_AMR_container_objects_train.joblib",
        "target_file": f"cache/masked_sentences_with_AMR_container_objects_{split}.joblib",
        "output_file": path,
    }
    if args.debug:
        main.calculate_empathy_similarities(
            source_file=config["source_file"],
            target_file=config["target_file"],
            source_feature=config["source_feature"],
            output_file=config["output_file"]
        )

    envs = "--export=ALL,"
    for key, value in config.items():
        envs += f"{key}={value},"

    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for {split} split")
    logger.info(
        f"Submitted job {job_id} for task {args.task} for split {split}")


def do_simcse_similarity(args, split, path):
    config = {
        "task": args.task,
        "source_feature": args.source_feature,
        "source_file": "cache/masked_sentences_with_AMR_container_objects_train.joblib",
        "target_file": f"cache/masked_sentences_with_AMR_container_objects_{split}.joblib",
        "output_file": path,
    }
    envs = "--export=ALL,"
    for key, value in config.items():
        envs += f"{key}={value},"

    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for {split} split")
    logger.info(
        f"Submitted job {job_id} for task {args.task} for split {split}")


def do_gcn_similarity(args):
    config = {
        "task": args.task,
        "source_feature": args.source_feature,
        "mid_layer_dropout": 0.3,
        "train_input_file": "cache/masked_sentences_with_AMR_container_objects_train.joblib",
        "dev_input_file": "cache/masked_sentences_with_AMR_container_objects_dev.joblib",
        "test_input_file": "cache/masked_sentences_with_AMR_container_objects_test.joblib",
        "batch_size": 8,
        "learning_rate": 5e-4,
        "num_epochs": 60,
        "g_type": "directed",
        "gcn_model_path": "cache/gcn_model.pt",
        "gcn_layers": [64, 64, 64]
    }
    if args.debug:
        main.gcn_similarity(config)
    envs = "--export=ALL,"
    for key, value in config.items():
        if type(value) == list:
            value = "&".join([str(v) for v in value])
        envs += f"{key}={value},"

    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for task {args.task} with args {args}")
    logger.info(
        f"Submitted job {job_id} for task {args.task} with args {args}")


def do_train_gcn(args):
    config = {
        "task": args.task,
        "source_feature": args.source_feature,
        "mid_layer_dropout": 0.3,
        "train_input_file": "cache/masked_sentences_with_AMR_container_objects_train.joblib",
        "dev_input_file": "cache/masked_sentences_with_AMR_container_objects_dev.joblib",
        "test_input_file": "cache/masked_sentences_with_AMR_container_objects_test.joblib",
        "batch_size": 8,
        "learning_rate": 5e-4,
        "num_epochs": 60,
        "g_type": "directed",
        "gcn_model_path": "cache/gcn_model.pt",
        "gcn_layers": [64, 64, 64]
    }
    if args.debug:
        main.train_gcn(config)
    envs = "--export=ALL,"
    for key, value in config.items():
        if type(value) == list:
            value = "&".join([str(v) for v in value])
        envs += f"{key}={value},"

    job_id = subprocess.check_output([
        'sbatch',
        envs,
        'slurm_job_scripts/general.sh'
    ]).decode()
    print(f"Submitted job {job_id} for task {args.task} with args {args}")
    logger.info(
        f"Submitted job {job_id} for task {args.task} with args {args}")


def do_train_main_classifier(args, path):
    config = {
        "source_feature": args.source_feature,
        "task": args.task,
        "train_input_file": "cache/masked_sentences_with_AMR_container_objects_train.joblib",
        "dev_input_file": "cache/masked_sentences_with_AMR_container_objects_dev.joblib",
        "test_input_file": "cache/masked_sentences_with_AMR_container_objects_test.joblib",
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sweep": args.sweep,
        "num_epochs": args.num_epochs,
        "augments": [],
        "retriever_type": args.retriever_type,
        "cbr": args.cbr,
        "num_cases": 1,
        "similarity_matrices_path_train": f"cache/simcse_similarities_{args.source_feature}_train.joblib",
        "similarity_matrices_path_dev": f"cache/simcse_similarities_{args.source_feature}_dev.joblib",
        "similarity_matrices_path_test": f"cache/simcse_similarities_{args.source_feature}_test.joblib",
        "classifier_dropout": 0.1,
        "cbr_threshold": args.cbr_threshold,
        "weight_decay": 0.001,
        "checkpoint": "roberta-base",
        "predictions_path": path
    }
    if args.debug:
        main.train_main_classifier(config)
    envs = "--export=ALL,"
    for key, value in config.items():
        envs += f"{key}={value},"

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
            print(
                f"AMR graphs for the {split} split already exist, skipping step 1")
            logger.info(
                f"AMR graphs for the {split} split already exist, skipping step 1")
        else:
            do_amr_generation(args, split, path)

    # second step is to train or prepare the case retrievers
    # First would be transformers retriever (both the simcse and sentence transformers)

    # There are two kinds of transformer retrievers we can use: simcse and sentence transformers
    # Also there are two ways we can do the training: use the masked sentences or use the original sentences

    # simcse similarity

    for split in ["train", "dev", "test"]:
        path = f"cache/simcse_similarities_{args.source_feature}_{split}.joblib"
        if os.path.exists(path):
            print(
                f"simcse similarities for the {split} split already exist, skipping step 2")
            logger.info(
                f"simcse similarities for the {split} split already exist, skipping step 2")
        else:
            do_simcse_similarity(args, split, path)

    # empathy similarity

    for split in ["train", "dev", "test"]:
        path = f"cache/empathy_similarities_{args.source_feature}_{split}.joblib"
        if os.path.exists(path):
            print(
                f"empathy similarities for the {split} split already exist, skipping step 2")
            logger.info(
                f"empathy similarities for the {split} split already exist, skipping step 2")
        else:
            do_empathy_similarity(args, split, path)

    # Third step is to train the case base reasoner
    do_train_main_classifier(args, args.predictions_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_feature', choices=[
                        'masked_articles', 'source_article', 'amr_str'], type=str, default='masked_articles')

    parser.add_argument('--task', type=str, choices=['amr_generation', 'simcse_similarity', "load_gcn",
                        'train_main_classifier', "train_gcn", "empathy_similarity", "reason", 'gcn_similarity'], default='follow_the_usual_process')

    parser.add_argument('--cbr', type=bool, default=False)
    parser.add_argument('--retriever_type', type=str,
                        choices=['simcse', 'empathy', 'gcn'], default="simcse")
    parser.add_argument(
        '--cbr_threshold', type=float, default=-1e9
    )
    parser.add_argument('--predictions_path', type=str)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument(
        '--debug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sweep', action=argparse.BooleanOptionalAction,
                        help="Whether to sweep over the hyperparameters", default=False)

    # Check all the arguments to be correct in the consts.py file because they are used in the whole project
    args = parser.parse_args()

    if args.task == "follow_the_usual_process":
        follow_the_usual_process(args)

    if args.task == "train_gcn":
        do_train_gcn(args)

    if args.task == "simcse_similarity":
        for split in ["train", "dev", "test"]:
            path = f"cache/simcse_similarities_{args.source_feature}_{split}.joblib"
            if os.path.exists(path):
                print(
                    f"simcse similarities for the {split} split already exist")
                logger.info(
                    f"simcse similarities for the {split} split already exist")
            else:
                do_simcse_similarity(args, split, path)

    if args.task == "empathy_similarity":
        for split in ["train", "dev", "test"]:
            path = f"cache/empathy_similarities_{args.source_feature}_{split}.joblib"
            if os.path.exists(path):
                print(
                    f"empathy similarities for the {split} split already exist")
                logger.info(
                    f"empathy similarities for the {split} split already exist")
            else:
                do_empathy_similarity(args, split, path)

    if args.task == "load_gcn":
        do_load_gcn(args)

    if args.task == "reason":
        do_train_main_classifier(args, args.predictions_path)

    if args.task == "gcn_similarity":
        do_gcn_similarity(args)
