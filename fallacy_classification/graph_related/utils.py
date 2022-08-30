from typing import Any, List
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from IPython import embed

from amr_container import AMR_Container


PATH_TO_TRAIN_DATA = ((Path(__file__).parent) /
                      "data/edu_train_amr_parse_tree.txt").absolute()


def graph2vec(list_of_graphs: List[nx.DiGraph]) -> None:
    """
    Generate the graph features over a set of graphs

    Args:
        list_of_graphs (List[Any]): list of graphs 
    """
    raise NotImplementedError()


def get_edgelist(graph: nx.DiGraph):
    return [x.split() for x in nx.generate_edgelist(graph, data=False)]


def calc_similarity(graph_1: Any, graph_2: Any) -> float:
    """
    return the similarity of two graphs

    Args:
        graph_1 (_type_): first graph
        graph_2 (_type_): second graph

    Returns:
        float: calculated similarity
    """
    raise NotImplementedError()


def get_amr_sentences_boundaries(lines: List[str]):
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
    return results


def read_amr_graph(path: Path) -> List[AMR_Container]:
    with open(path, 'r') as f:
        lines = f.readlines()

    amr_sentences_boundaries = get_amr_sentences_boundaries(
        lines
    )
    amr_sentences = [
        ''.join(lines[start:end])
        for start, end
        in amr_sentences_boundaries
    ]

    graphs = []
    n = 0
    for amr_sentence in amr_sentences:
        try:
            graphs.append(
                AMR_Container(
                    graph_str=amr_sentence
                )
            )
        except Exception as e:
            n += 1
            continue
    print(n)
    print(len(amr_sentences))


if __name__ == "__main__":
    read_amr_graph(path=PATH_TO_TRAIN_DATA)
