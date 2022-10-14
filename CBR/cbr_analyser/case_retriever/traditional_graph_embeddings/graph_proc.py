import json
import os
from typing import Any, List

import networkx as nx
import pandas as pd
import tqdm
from sklearn.metrics import pairwise_distances

from cbr_analyser.consts import *


def get_edgelist(graph: nx.DiGraph):
    label2int = {}
    for index, node in enumerate(graph.nodes()):
        if node not in label2int:
            label2int[node] = index
    edge_list =  [x.split() for x in nx.generate_edgelist(graph, data=False)]
    edge_list = [
        [
            label2int[pair[0]],
            label2int[pair[1]]
        ]
        for pair 
        in edge_list
    ]
    return edge_list



def generate_all_edge_lists():
    results = get_amr_labels_from_csv_file(csv_path='data/edu_train.csv')
    for index, result in tqdm(enumerate(results), leave = False):
        digraph = result[1].graph_nx
        edge_list = get_edgelist(digraph)
        with open(os.path.join("tmp", "edge_lists", f"{index}.json"), 'w') as f:
            json.dump({
                "edges": edge_list
                }, f)

def get_graph_similarity_gmatch(graph_a: nx.Graph, graph_b: nx.Graph) -> float:
    raise NotImplementedError()


def get_similar_graphs_graph2vec(
    index,
    graph_embeddings: pd.DataFrame,
    sentences_with_amr_container: List[Any],
):
    kernel_name = "graph2vec"
    graph_embeddings.set_index('type', inplace = True)
    graph_embeddings_indices = graph_embeddings.index.tolist()
    dist_out = 1-pairwise_distances(graph_embeddings.values, metric="cosine")
    similarity_matrix_df = pd.DataFrame(
        dist_out,
        index=graph_embeddings_indices,
        columns=graph_embeddings_indices
    )
    
    results = pd.DataFrame()

    for index in similarity_matrix_df.index:
        most_similar_graphs = similarity_matrix_df.loc[index].nlargest(20).index.tolist()
        most_similar_graphs_scores = similarity_matrix_df.loc[index].nlargest(20).values
        for similar_index, similarity_score in zip(most_similar_graphs, most_similar_graphs_scores):
            if similar_index != index:
                results = results.append({
                    'sent_a': sentences_with_amr_container[index][1].sentence,
                    'sent_b': sentences_with_amr_container[similar_index][1].sentence,
                    'type_a': sentences_with_amr_container[index][2],
                    'type_b': sentences_with_amr_container[similar_index][2],
                    'similarity': similarity_score
                }, ignore_index=True)
    
    
    results.to_csv(f"{PATH_TO_MOST_SIMILAR_GRAPHS}{kernel_name}.csv", index = False)




if __name__ == "__main__":
    pass

    