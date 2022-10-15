import argparse
import os
import sys

import joblib
from IPython import embed

from cbr_analyser.amr.amr_extraction import (
    augment_amr_container_objects_with_clean_node_labels,
    generate_amr_containers_from_csv_file)
from cbr_analyser.case_retriever.transformers.simcse_similarity_calculations import \
    generate_the_simcse_similarities
from cbr_analyser.consts import *
from cbr_analyser.reasoner.main_classifier import do_train_process

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "cbr_analyser/amr/"))


def generate_amr(input_file, output_file):
    generate_amr_containers_from_csv_file(
        input_data_path=input_file,
        output_data_path=output_file
    )

    augment_amr_container_objects_with_clean_node_labels(
        sentences_with_amr_container=joblib.load(output_file),
        output_path=output_file
    )


def calculate_simcse_similarities(source_file, source_feature, target_file, output_file):
    generate_the_simcse_similarities(
        source_feature=source_feature,
        source_file=source_file,
        target_file=target_file,
        output_file=output_file
    )


def train_main_classifier(args):
    do_train_process(vars(args))


if __name__ == "__main__":
    print('begin')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help="input file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--output_file', help="output file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument(
        '--task', help="The task that should be done", type=lambda x: None if str(x) == "default" else str(x), choices=['amr_generation', 'simcse_similarity', 'train_main_classifier'])
    parser.add_argument("--source_feature",
                        help="The source feature that should be used")

    parser.add_argument('--source_file', help="The source file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--target_file', help="The target file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--train_input_file',
                        help="The train input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--dev_input_file',
                        help="The dev input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--test_input_file',
                        help="The test input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--batch_size', help="The batch size", type=int)
    parser.add_argument('--learning_rate',
                        help="The learning rate", type=float)
    parser.add_argument('--num_epochs', help="The number of epochs", type=int)
    parser.add_argument('--classifier_dropout',
                        help="The classifier dropout", type=float)
    parser.add_argument('--weight_decay', help="The weight decay", type=float)
    parser.add_argument('--augments', help="The augments",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument(
        '--cbr', help="Whether the cbr is enabled or not", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--similarity_matrices_path_train',
                        help="The similarity matrices path train", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--similarity_matrices_path_dev',
                        help="The similarity matrices path dev", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--similarity_matrices_path_test',
                        help="The similarity matrices path test", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--num_cases', help="The number of cases", type=int)
    parser.add_argument('--all_good_cases',
                        help="path to the good cases for case base", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--all_bad_cases',
                        help="path to the bad cases for case base", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--cbr_threshold',
                        help="The cbr threshold", type=float)
    parser.add_argument(
        '--checkpoint', help="The checkpoint to load the model from", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument(
        '--predictions_path', help="The path to the predictions", type=lambda x: None if str(x) == "default" else str(x))

    args = parser.parse_args()

    print(args)

    if args.task == "amr_generation":
        generate_amr(args.input_file, args.output_file)

    elif args.task == "simcse_similarity":
        calculate_simcse_similarities(
            source_file=args.source_file,
            source_feature=args.source_feature,
            target_file=args.target_file,
            output_file=args.output_file
        )
    elif args.task == "train_main_classifier":
        train_main_classifier(args)
