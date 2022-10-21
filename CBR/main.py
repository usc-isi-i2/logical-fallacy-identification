import argparse
import os
import sys
import wandb
from typing import Any, Dict

import joblib
from IPython import embed

from cbr_analyser.amr.amr_extraction import (
    augment_amr_container_objects_with_clean_node_labels,
    generate_amr_containers_from_csv_file)
from cbr_analyser.case_retriever.gcn import gcn
from cbr_analyser.case_retriever.retriever import GCN_Retriever
from cbr_analyser.consts import *
from cbr_analyser.reasoner.main_classifier import do_train_process

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "cbr_analyser/amr/"))


def train_gcn(args: Dict[str, Any]):
    gcn.train(args)


def gcn_similarity(args: Dict[str, Any]):
    gcn.get_similarities(args)


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
    from cbr_analyser.case_retriever.transformers.simcse_similarity_calculations import \
        generate_the_simcse_similarities
    generate_the_simcse_similarities(
        source_feature=source_feature,
        source_file=source_file,
        target_file=target_file,
        output_file=output_file
    )


def calculate_empathy_similarities(source_file, source_feature, target_file, output_file):
    from cbr_analyser.case_retriever.transformers.empathy_similarity_calculations import \
        generate_the_empathy_similarities
    generate_the_empathy_similarities(
        source_feature=source_feature,
        source_file=source_file,
        target_file=target_file,
        output_file=output_file
    )


def train_wrapper(config=None):
    with wandb.init(config=config):
        args = wandb.config
        for metrics in do_train_process(args):
            wandb.log(metrics, step=metrics["epoch"])


def train_main_classifier(args: Dict[str, Any]):
    if args["sweep"] == True:
        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'valid_f1',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            'learning_rate': {
                'distribution': 'uniform',
                'min': 1e-6,
                'max': 2e-5
            },
            'num_epochs': {
                'value': 20
            },
            'weight_decay': {
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.001
            },
            "encoder_dropout_rate": {
                'values': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "attn_dropout_rate": {
                'values': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "last_layer_dropout": {
                'values': [0.1, 0.2, 0.3, 0.4]
            }
        }
        for key, value in args.items():
            if key not in parameters_dict:
                parameters_dict.update(
                    {
                        key: {'value': value}
                    }
                )

        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(
            sweep_config, project="Sweep for main classifier CBR")
        wandb.agent(sweep_id, train_wrapper, count=50)
    else:
        do_train_process(args)


def load_gcn(args):
    print('starting to load!!')
    retriever = GCN_Retriever(
        gcn_model_path="cache/gcn_model.pt",
        config={
            "gcn_layers": [128, 64, 32],
            "mid_layer_dropout": 0.5
        },
        train_input_file="cache/masked_sentences_with_AMR_container_objects_train.joblib"
    )
    print('loaded!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help="input file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--output_file', help="output file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument(
        '--task', help="The task that should be done", type=lambda x: None if str(x) == "default" else str(x), choices=['amr_generation', 'simcse_similarity', 'train_gcn', 'empathy_similarity', "load_gcn", 'reason', 'gcn_similarity'])
    parser.add_argument("--source_feature", type=lambda x: "masked_articles" if str(x) == "default" else str(x),
                        help="The source feature that should be used", choices=['masked_articles', 'source_article', 'amr_str'])

    parser.add_argument("--sweep", help="Whether to do a sweep",
                        type=lambda x: False if str(x) == "default" else str(x.lower()) == 'true')
    parser.add_argument('--source_file', help="The source file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--g_type', help="The type of the graph",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--target_file', help="The target file",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--train_input_file',
                        help="The train input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--dev_input_file',
                        help="The dev input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--test_input_file',
                        help="The test input file", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--batch_size', help="The batch size",
                        type=lambda x: None if str(x) == "default" else int(x))
    parser.add_argument('--learning_rate',
                        help="The learning rate", type=lambda x: None if str(x) == "default" else float(x))
    parser.add_argument('--num_epochs', help="The number of epochs",
                        type=lambda x: None if str(x) == "default" else int(x))
    parser.add_argument('--classifier_dropout',
                        help="The classifier dropout", type=lambda x: None if str(x) == "default" else float(x))
    parser.add_argument('--mid_layer_dropout',
                        help="The mid layer dropout", type=lambda x: None if str(x) == "default" else float(x))
    parser.add_argument('--gcn_model_path', help="The gcn model path",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--gcn_layers', help="The number of gcn neurons in each layer (separated by comma)",
                        type=lambda x: [] if str(x) == "default" else [int(item) for item in x.split('&')])
    parser.add_argument('--weight_decay', help="The weight decay",
                        type=lambda x: None if str(x) == "default" else float(x))
    parser.add_argument('--retriever_type', help="The retriever type",
                        type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--augments', help="The augments",
                        type=lambda x: [] if str(x) == "default" else x.split('&'))
    parser.add_argument(
        '--cbr', help="Whether the cbr is enabled or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--similarity_matrices_path_train',
                        help="The similarity matrices path train", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--similarity_matrices_path_dev',
                        help="The similarity matrices path dev", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--similarity_matrices_path_test',
                        help="The similarity matrices path test", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--num_cases', help="The number of cases",
                        type=lambda x: None if str(x) == "default" else int(x))
    parser.add_argument('--all_good_cases',
                        help="path to the good cases for case base", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--all_bad_cases',
                        help="path to the bad cases for case base", type=lambda x: None if str(x) == "default" else str(x))
    parser.add_argument('--cbr_threshold',
                        help="The cbr threshold", type=lambda x: None if str(x) == "default" else float(x))
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
    elif args.task == "reason":
        train_main_classifier(vars(args))

    elif args.task == "train_gcn":
        train_gcn(vars(args))

    elif args.task == "empathy_similarity":
        calculate_empathy_similarities(
            source_file=args.source_file,
            source_feature=args.source_feature,
            target_file=args.target_file,
            output_file=args.output_file
        )

    elif args.task == "load_gcn":
        load_gcn(vars(args))

    elif args.task == "gcn_similarity":
        gcn_similarity(vars(args))
