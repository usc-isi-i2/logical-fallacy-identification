import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from IPython import embed
from tqdm import tqdm

from cbr_analyser.amr.amr_container import AMR_Container

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "./"))
from cbr_analyser.custom_logging.custom_logger import get_logger

warnings.filterwarnings("ignore")
import argparse

logger = get_logger(
        logger_name=f"{__name__}.{os.path.basename(__file__)}"
    )

def get_amr_sentences_from_amr_generated_lines(lines: List[str]):
    results = []
    i = 0
    while i != len(lines):
        if lines[i].startswith("# ::snt"):
            start = i
            i += 1
            while lines[i] != '\n':
                i += 1
            results.append([start, i])
        else:
            i += 1
    amr_sentences = [
        ''.join(lines[start:end])
        for start, end
        in results
    ]
    return amr_sentences


def get_raw_sentences_from_amr_generated_lines(lines: List[str]):
    results = []
    i = 0
    while i != len(lines):
        if lines[i].startswith("# ::snt"):
            start = i
            i += 1
            while not re.match(r'\([\S]+ \/ [\S]+\n', lines[i]):
                i += 1
            results.append([start, i])
        else:
            i += 1
    raw_sentences = [
        ''.join(lines[start:end])
        for start, end
        in results
    ]
    return raw_sentences

def generate_amr_containers_from_csv_file(input_data_path: Path or str, output_data_path: Path or str) -> None:
    """
    Read the csv data containing the source article, masked_article, and updated labels (train, dev, or test),
    and create the AMR Container instances for them containg the AMR graphs in Graphviz and Networkx format. 

    Args:
        input_data_path (Pathorstr): input data path
        output_data_path (Pathorstr): path to save the outputs

    Returns:
        _type_: None
    """
    if os.path.exists(output_data_path):
        results = joblib.load(output_data_path)
        return results

    df = pd.read_csv(input_data_path)

    results = []
    

    for _, (_, data) in tqdm(enumerate(df.iterrows()), leave=False):
        try:

            label = data["label"]
            # masked_article = data["masked_articles"]
            # original_article = data["source_article"]
            masked_article = data["text"]
            original_article = data["text"]

            updated_masked_article = re.sub(
                r"MSK<(\d+)>", r"MSK\1", masked_article
            )

            updated_masked_article = re.sub(r"\n", ". ", updated_masked_article)

            amr_container = AMR_Container(
                sentence=updated_masked_article
            )
            results.append([
                original_article,
                amr_container,
                label
            ])
        except Exception as e:
            logger.error(
                f"File: {input_data_path}, Error handling data: {updated_masked_article} with the following error: {e} \n"
            )
            continue
    joblib.dump(
        results,
        output_data_path
    )

def get_clearn_node_labels_for_graph(g: nx.DiGraph):
    label2word = {}
    for node in g.nodes(data=True):
        try:
            node_name = node[0]
            label = node[1]['label']
            if label == '"-"':
                label = '"negative"'
            if label == '"+"':
                label = '"positive"'
            if not re.match(r'".*"', label):
                label = f'"{label}"'

            if '/' in label and '-' in label:
                pattern = r'"[a-zA-Z0-9]+/([a-zA-Z-]+)(-\d*)?"'
                word = re.findall(pattern, label)[0][0]
                word = re.sub('-', ' ', word)
            elif '/' in label and '-' not in label:
                pattern = r'"[a-zA-Z0-9]+/([\w]+)"'
                word = re.findall(pattern, label)[0]

            else:
                word = re.findall(r'"(.*)"', label)[0]

            if word == " ":
                print(label)
                embed()
        except Exception as e:
            logger.error(
                f"Error finding the clean label of: {label} with the following error: {e} \n"
            )
            word = label
            
        label2word[node_name] = word
    return label2word


def augment_amr_container_objects_with_clean_node_labels(sentences_with_amr_container, output_path: str or Path) -> None:
    for obj in tqdm(sentences_with_amr_container, leave = False):
        try:
            graph = obj[1].graph_nx
            label2word = get_clearn_node_labels_for_graph(g = graph)
            obj[1].add_label2word(label2word)

        except Exception as e:
            obj[1].add_label2word(None)
            logger.error(
                f"Error augmenting the graph of sentence: {obj[1].sentence} with the following error: {e} \n"
            )
            continue
    
    joblib.dump(
        sentences_with_amr_container,
        output_path
    )

    
# def compute_statistics_based_on_different_kernels():
#     kernel_names = ["edge_histogram", "graph2vec", 'graphlet_sampling', 'transformers']
#     statistics = pd.DataFrame()

#     top_n = 10

#     for kernel_name in kernel_names:
#         results = pd.read_csv(f"{PATH_TO_MOST_SIMILAR_GRAPHS}{kernel_name}.csv")
#         fallacy_types_counts = results[['sent_a', 'type_a']].drop_duplicates(
#         ).groupby("type_a").apply(lambda x: len(x))
        
#         all_types = results['type_a'].unique().tolist()
#         for type in all_types:
#             type_records = results[results['type_a'] == type]
#             type_records = type_records.sort_values(by=['sent_a', 'similarity'])

#             sub_type_records = type_records.groupby('sent_a').apply(
#                 lambda x: x[:top_n]).reset_index(drop=True)

#             num_sentences = sub_type_records['sent_a'].nunique()
#             match_for_each_sentence_vec = sub_type_records.groupby('sent_a').apply(
#                 lambda x: (np.array(x['type_b'].tolist()) == np.array([type])).astype(int)).values

#             statistics = statistics.append({
#                 'algorithm': kernel_name,
#                 'type': type,
#                 'num_records': num_sentences,
#                 'top_n': top_n,
#                 'ratio/all_classes': num_sentences / np.sum(fallacy_types_counts),
#                 'MAP': mean_average_precision(match_for_each_sentence_vec),
#                 'random_baseline_MAP': compute_random_baseline(fallacy_types_counts, type, top_n)
#             }, ignore_index=True)

#     statistics.to_csv(PATH_TO_STATISTICS, index=False)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The basic stuff to do with the sentences and their AMR graphs")
    parser.add_argument('--input_file', help="input file")
    parser.add_argument('--output_file', help = "output file")
    parser.add_argument('--task', help = "The task that should be done")
    args = parser.parse_args()


    logger = get_logger(
        logger_name=f"{__name__}.{os.path.basename(__file__)}"
    )

    
    if args.task == "amr_generation":
        generate_amr_containers_from_csv_file(
            input_data_path=args.input_file,
            output_data_path=args.output_file
        )

        augment_amr_container_objects_with_clean_node_labels(
            sentences_with_amr_container=joblib.load(args.output_file),
            output_path=args.output_file
        )
                