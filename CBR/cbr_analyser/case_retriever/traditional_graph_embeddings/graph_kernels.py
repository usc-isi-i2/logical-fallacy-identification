import re
import warnings

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from grakel import Graph, GraphKernel
from IPython import embed
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from cbr_analyser.augmentations.embedding_extractor import get_bert_embeddings
from cbr_analyser.consts import (PATH_TO_MASKED_SENTENCES_AMRS,
                                 PATH_TO_MOST_SIMILAR_GRAPHS)

warnings.filterwarnings("ignore")

model = None


def get_label_mappings(amr_graphs_with_sentences):
    label2index = {}
    all_labels = set()
    for obj in amr_graphs_with_sentences:
        all_labels.add(obj[2])

    label2index = {
        label: index
        for index, label
        in enumerate(all_labels)
    }

    return label2index


def get_node_mappings(g):
    node2index = {}
    for i, node in enumerate(g.nodes()):
        node2index[node] = i
    return node2index


def get_node_embeddings(g):
    label2embeddings = {}
    pattern = r'"[a-zA-Z0-9]+/([\w]+)(-\d*)?"'

    for node in g.nodes(data=True):
        label = node[1]['label']
        word = re.findall(pattern, label)[0][0]

        label2embeddings[label] = get_bert_embeddings(word)
    return label2embeddings


def get_edge_mappings(amr_graphs_with_sentences):
    edge_type2index = {}
    all_edge_types = set()
    for obj in amr_graphs_with_sentences:
        g = obj[1].graph_nx
        for edge in g.edges(data=True):
            arg = edge[2]['label']
            all_edge_types.add(arg)

    edge_type2index = {
        edge_type: index
        for index, edge_type
        in enumerate(all_edge_types)
    }

    return edge_type2index


def from_networkx(g: nx.Graph or nx.Digraph, edge_type2index) -> Graph:
    edges = {}
    edge_labels = {}
    node2index = get_node_mappings(g=g)
    for edge in g.edges(data=True):
        a = node2index[edge[0]]
        b = node2index[edge[1]]
        if (a, b) not in edges:
            edges[(a, b)] = 1
            edge_labels[(a, b)] = edge_type2index[edge[2]['label']]
        if (b, a) not in edges:
            edges[(b, a)] = 1
            edge_labels[(b, a)] = edge_type2index[edge[2]['label']]
    return Graph(edges, edge_labels=edge_labels)


def get_graph_similarities_by_sentence_transformers(amr_graphs_with_sentences):
    global model
    if model is None:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings = []
    for obj in tqdm(amr_graphs_with_sentences, leave=False):
        sentence = [obj[1].sentence]
        embedding = model.encode(sentence)
        embeddings.append(embedding.squeeze())

    embeddings = np.array(embeddings)
    dist_out = 1-pairwise_distances(embeddings, metric="cosine")

    similarity_matrix_df = pd.DataFrame(
        dist_out,
        index=np.arange(len(amr_graphs_with_sentences)),
        columns=np.arange(len(amr_graphs_with_sentences))
    )

    results = pd.DataFrame()

    for index in similarity_matrix_df.index:
        most_similar_graphs = similarity_matrix_df.loc[index].nlargest(
            10).index.tolist()
        most_similar_graphs_scores = similarity_matrix_df.loc[index].nlargest(
            10).values
        for similar_index, similarity_score in zip(most_similar_graphs, most_similar_graphs_scores):
            if similar_index != index:
                results = results.append({
                    'sent_a': amr_graphs_with_sentences[index][1].sentence,
                    'sent_b': amr_graphs_with_sentences[similar_index][1].sentence,
                    'type_a': amr_graphs_with_sentences[index][2],
                    'type_b': amr_graphs_with_sentences[similar_index][2],
                    'similarity': similarity_score
                }, ignore_index=True)

    results.to_csv(
        f"{PATH_TO_MOST_SIMILAR_GRAPHS}{'transformers'}.csv", index=False)


def compute_graph_similarity_using_graph_kernel(kernel_name):
    amr_graphs_with_sentences = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS)
    edge_type2index = get_edge_mappings(
        amr_graphs_with_sentences=amr_graphs_with_sentences)

    grakel_graphs = []
    for obj in amr_graphs_with_sentences:
        g = obj[1].graph_nx
        grakel_graphs.append(
            from_networkx(
                g=g,
                edge_type2index=edge_type2index
            )
        )

    kernel = GraphKernel(kernel=kernel_name, normalize=True)

    results = pd.DataFrame()

    for i in tqdm(range(len(grakel_graphs)), leave=False):
        amr_graph_with_sentence = amr_graphs_with_sentences[i]
        grakel_graph = grakel_graphs[i]
        kernel = GraphKernel(kernel=kernel_name, normalize=True)
        kernel.fit_transform([grakel_graph])

        similarities = kernel.transform(grakel_graphs).squeeze()
        embed()
        similarities = [similarity if not np.isnan(
            similarity) else 0 for similarity in similarities]

        similar_indices = np.argpartition(similarities, -11)[-11:]
        similar_indices = [index for index in similar_indices if index != i]

        for index in similar_indices:
            sent_a = amr_graph_with_sentence[1].sentence
            type_a = amr_graph_with_sentence[2]
            sent_b = amr_graphs_with_sentences[index][1].sentence
            type_b = amr_graphs_with_sentences[index][2]
            similarity = similarities[index]

            results = pd.concat([results, pd.DataFrame({
                'sent_a': [sent_a],
                'sent_b': [sent_b],
                'type_a': [type_a],
                'type_b': [type_b],
                'similarity': [similarity]
            })], axis=0, ignore_index=True)

    results.to_csv(
        f"{PATH_TO_MOST_SIMILAR_GRAPHS}{kernel_name}.csv", index=False)


if __name__ == "__main__":
    amr_graphs_with_sentences = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS)
    get_graph_similarities_by_sentence_transformers(
        amr_graphs_with_sentences=amr_graphs_with_sentences
    )
