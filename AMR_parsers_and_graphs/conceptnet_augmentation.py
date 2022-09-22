from collections import defaultdict
import requests
from custom_logger import get_logger
import nltk
import joblib
from consts import PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET, \
    PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET_CONCEPTNET
import numpy as np
import os
from tqdm import tqdm
import wandb
import argparse

parser = argparse.ArgumentParser(
    description='Augment the AMR graphs with different Knowledge Bases')

parser.add_argument(
    '--input_file', help="The path to the input file")
parser.add_argument('--output_file', help="The path to the output file")

args = parser.parse_args()

pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV']

logger = get_logger(f"{__name__}.{os.path.basename(__file__)}")
wandb.init(project=f"conceptnet_augmentation", entity="zhpinkman")


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
                'edge': (edge['start']['label'], edge['rel']['label'], edge['end']['label']),
                'example': edge['surfaceText']
            }


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
    cached_edges = dict()

    sentences_with_amr_container = joblib.load(args.input_file)
    print(f"Number of records {len(sentences_with_amr_container)}")
    for obj in tqdm(sentences_with_amr_container, leave=False):
        try:
            graph = obj[1].graph_nx
            label2word = obj[1].label2word
            graph_edges = graph.edges(data=False)
            graph_nodes = graph.nodes(data=False)

            graph_stats = defaultdict(int)

            all_hits = 0
            all_misses = 0

            # print(len(graph_nodes))
            for node1 in graph_nodes:
                for node2 in graph_nodes:
                    node1_label = label2word[node1].lower()
                    node2_label = label2word[node2].lower()

                    if nltk.pos_tag([node1_label], tagset='universal')[0][1] not in pos_tags or \
                            nltk.pos_tag([node2_label], tagset='universal')[0][1] not in pos_tags or \
                            node1_label.startswith('msk') or node2_label.startswith('msk'):
                        continue

                    if node1_label != node2_label and (node1, node2) not in graph_edges and (node2, node1) not in graph_edges:

                        if (node1_label, node2_label) in cached_edges:
                            all_hits += 1
                            for cached_edge in cached_edges[(node1_label, node2_label)]:
                                graph.add_edge(
                                    node1, node2, label=cached_edge[0], example=cached_edge[1])
                                graph_stats[cached_edge[0]] += 1
                        else:
                            # print(node1_label, node2_label)
                            all_misses += 1
                            cached_edges[(node1_label, node2_label)] = []
                            for new_edge in get_relations_between_words(node1_label, node2_label):
                                graph.add_edge(
                                    node1, node2, label=new_edge['edge'][1], example=new_edge['example'])
                                cached_edges[(node1_label, node2_label)].append(
                                    (new_edge['edge'][1], new_edge['example']))
                                graph_stats[new_edge['edge'][1]] += 1
                                print('...')

            print(all_misses)

            wandb.log({'hit/all ratio': all_hits / (all_hits + all_misses)})

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
