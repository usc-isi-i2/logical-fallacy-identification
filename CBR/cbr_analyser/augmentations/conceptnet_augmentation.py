import argparse
import os
from collections import defaultdict

import joblib
import nltk
import numpy as np
import requests
from IPython import embed
from tqdm import tqdm

import cbr_analyser.consts as consts
import wandb
from cbr_analyser.custom_logging.custom_logger import get_logger

parser = argparse.ArgumentParser(
    description='Augment the AMR graphs with different Knowledge Bases')

parser.add_argument(
    '--input_file', help="The path to the input file")
parser.add_argument('--output_file', help="The path to the output file")

parser.add_argument('--rel_file', help="path of the relations inverted index")
parser.add_argument('--label_file', help="path of the labels of the nodes")
parser.add_argument('--good_relations', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV']

logger = get_logger(f"{__name__}.{os.path.basename(__file__)}")
wandb.init(project=f"conceptnet_augmentation", entity="zhpinkman")


rel_dict = joblib.load(args.rel_file)
label_dict = joblib.load(args.label_file)


def make_request(query):
    base_url = "http://api.conceptnet.io"
    if not query.startswith(base_url):
        query = f"{base_url}{query}"
    response = requests.get(query)
    return response.json()


def read_edges(obj):
    for edge in obj['edges']:
        if edge['start']['language'] == 'en' and edge['end']['language'] == 'en':
            yield {
                'start': edge['start']['label'].lower(),
                'start_id': edge['start']['@id'].lower(),
                'end': edge['end']['label'].lower(),
                'end_id': edge['end']['@id'].lower(),
                'rel': edge['rel']['label'],
                'example': edge['surfaceText']
            }


def get_relations_between_words_from_dump(word1, word2):
    word1 = word1.replace(' ', '_')
    word2 = word2.replace(' ', '_')

    edges = []

    for rel in rel_dict[(word1, word2)]:
        if args.good_relations:
            if rel[0] in consts.good_relations_labels:
                edges.append({
                    'start': label_dict[word1],
                    'rel': rel[0],
                    'end': label_dict[word2],
                    'example': rel[1]
                })
        else:
            edges.append({
                'start': label_dict[word1],
                'rel': rel[0],
                'end': label_dict[word2],
                'example': rel[1]
            })
    return edges


def get_relations_between_words(word1, word2):
    word1 = word1.replace(' ', '_')
    word2 = word2.replace(' ', '_')
    query = f"/query?node=/c/en/{word1}&other=/c/en/{word2}&language=en"
    obj = make_request(query=query)
    edges = list(read_edges(obj))
    return edges


def get_edges_related_to_word(word):
    word = word.replace(' ', '_')
    query = f"/query?node=/c/en/{word}&language=en"
    obj = make_request(query=query)
    edges = list(read_edges(obj))
    return edges


if __name__ == "__main__":

    statistics = defaultdict(list)

    sentences_with_amr_container = joblib.load(args.input_file)
    print(f"Number of records {len(sentences_with_amr_container)}")
    for obj in tqdm(sentences_with_amr_container, leave=False):
        try:
            graph = obj[1].graph_nx
            label2word = obj[1].label2word
            graph_edges = graph.edges(data=False)
            graph_nodes = graph.nodes(data=False)

            graph_stats = defaultdict(int)

            for node1 in graph_nodes:
                for node2 in graph_nodes:
                    node1_label = label2word[node1].lower()
                    node2_label = label2word[node2].lower()

                    if node1_label.startswith('msk') or node2_label.startswith('msk') or node1_label == node2_label:
                        continue
                    # print(node1_label, node2_label)
                    for new_edge in get_relations_between_words_from_dump(node1_label, node2_label):
                        graph.add_edge(
                            node1, node2, label=new_edge['rel'], example=new_edge['example'])

                        graph_stats[new_edge['rel']] += 1

            for edge_type, count in graph_stats.items():
                statistics[edge_type].append(count)

            for edge_type in graph_stats.keys():
                added_edges = [(label2word[edge[0]], label2word[edge[1]]) for edge in graph.edges(
                    data=True) if edge[2]['label'] == edge_type]
                if len(added_edges):
                    logger.info(
                        f"{edge_type}: {added_edges}"
                    )
        except Exception as e:
            logger.error(f"Error: {e}")
            continue

    for edge_type in statistics.keys():
        print(
            f"Number of edges added based on words being {edge_type}: {np.sum(statistics[edge_type]) / len(sentences_with_amr_container)}")

    joblib.dump(
        sentences_with_amr_container,
        args.output_file
    )
