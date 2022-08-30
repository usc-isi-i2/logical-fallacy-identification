from typing import Any, List
import networkx as nx
import joblib
import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from IPython import embed

from amr_container import AMR_Container


PATH_TO_TRAIN_DATA = ((Path(__file__).parent) /
                      "data/edu_train_amr_parse_tree.txt").absolute()


PATH_TO_SENTENCES_AMR_OBJECTS = ((Path(__file__).parent) /
                                 "tmp/sentences_with_AMR_container_objects.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMR_AMRS = ((Path(__file__).parent) /
                                     "tmp/masked_sentences_with_AMR_container_objects.joblib").absolute()


def graph2vec(list_of_graphs: List[nx.DiGraph]) -> None:
    # TODO
    """
    Generate the graph features over a set of graphs

    Args:
        list_of_graphs (List[Any]): list of graphs 
    """
    raise NotImplementedError()


def get_edgelist(graph: nx.DiGraph):
    return [x.split() for x in nx.generate_edgelist(graph, data=False)]


def calc_similarity(graph_1: Any, graph_2: Any) -> float:
    # TODO (also check the other similarities)
    """
    return the similarity of two graphs

    Args:
        graph_1 (_type_): first graph
        graph_2 (_type_): second graph

    Returns:
        float: calculated similarity
    """
    raise NotImplementedError()


def get_amr_sentences(lines: List[str]):
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


def get_raw_sentences(lines: List[str]):
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


def read_amr_graph(path: Path) -> List[AMR_Container]:
    with open(path, 'r') as f:
        lines = f.readlines()

    amr_sentences = get_amr_sentences(
        lines=lines
    )
    raw_sentences = get_raw_sentences(
        lines=lines
    )

    graphs = []
    n = 0
    for raw_sentence, amr_sentence in tqdm(zip(raw_sentences, amr_sentences), leave=False):
        try:
            # TODO: check why some cases fail to generate the graphviz instance
            # FIXME: the sentences which contain \n (related to the part that we have multiple sentences in one)
            graphs.append(
                AMR_Container(
                    sentence=raw_sentence,
                    graph_str=amr_sentence
                )
            )
        except Exception as e:
            n += 1
            continue
    print(n, len(raw_sentences), len(amr_sentences))
    return graphs


if __name__ == "__main__":
    # if os.path.exists(PATH_TO_SENTENCES_AMR_OBJECTS):
    #     graphs = joblib.load(PATH_TO_SENTENCES_AMR_OBJECTS)
    # else:
    #     graphs = read_amr_graph(path=PATH_TO_TRAIN_DATA)
    #     joblib.dump(
    #         graphs,
    #         PATH_TO_SENTENCES_AMR_OBJECTS
    #     )
    # embed()

    df = pd.read_csv('data/edu_train.csv')

    results = []

    for index, (row, data) in tqdm(enumerate(df.iterrows()), leave=False):

        label = data["updated_label"]
        masked_article = data["masked_articles"]
        original_article = data["source_article"]

        updated_masked_article = re.sub(
            r"MSK<(\d+)>", r"MSK\1", masked_article)

        updated_masked_article = re.sub(r"\n", " ", updated_masked_article)

        amr_container = AMR_Container(
            sentence=updated_masked_article
        )
        results.append([
            original_article,
            amr_container,
            label
        ])
    joblib.dump(
        results,
        PATH_TO_MASKED_SENTENCES_AMR_AMRS
    )
