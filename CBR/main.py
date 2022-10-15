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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help="input file", type=str)
    parser.add_argument('--output_file', help="output file", type=str)
    parser.add_argument(
        '--task', help="The task that should be done", type=str, choices=['amr_generation', 'simcse_similarity'])
    parser.add_argument("--source_feature",
                        help="The source feature that should be used")

    parser.add_argument('--source_file', help="The source file", type=str)
    parser.add_argument('--target_file', help="The target file", type=str)

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
