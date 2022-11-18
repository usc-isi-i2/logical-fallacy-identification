from typing import Dict

import amrlib
import graphviz  # sudo apt install graphviz; pip3 install graphviz
import joblib
import networkx
import networkx as nx
import pydotplus
from amrlib.graph_processing.amr_plot import *
from IPython import embed


class AMR_Container:
    """
    its instances will contain the AMR representation of the sentences. 
    https://github.com/bjascob/amrlib
    """

    stog_model = None

    def __init__(self, sentence: str = None, graph_str: str = None) -> None:

        self.sentence = sentence
        if graph_str is None:
            self.graph_str = self.generate_amr()
        else:
            self.graph_str = graph_str

        self.graph_viz = self.generate_graphviz()
        self.graph_nx = self.generate_networkx()

    def load_model(self) -> None:
        try:
            device = "cuda"
            AMR_Container.stog_model = amrlib.load_stog_model(device=device)
            print('using GPU')
        except Exception as e:
            print(e)
            print('using CPU')
            device = "cpu"
            AMR_Container.stog_model = amrlib.load_stog_model(device=device)

    def generate_amr(self) -> str:
        """
        Create the AMR representation of the sentence

        Returns:
            str: AMR representation of the sentence
        """
        if AMR_Container.stog_model is None:
            self.load_model()
        graph = self.stog_model.parse_sents([self.sentence])[0]
        return graph

    def generate_graphviz(self) -> graphviz.Digraph:
        """
        Get the graphviz representation of the graph

        Returns:
            graphviz.Digraph: The AMR representation of the graph in graphviz Directed graph format
        """
        amr_plot = AMRPlot()
        amr_plot.build_from_graph(self.graph_str)
        return amr_plot.graph

    def generate_networkx(self) -> nx.DiGraph:
        """
        Get the networkx representation of the graph

        Returns:
            nx.DiGraph: The AMR representation of the graph in networkx Directed graph format
        """
        source = self.graph_viz.source.replace("\\\"", "")
        dotplus = pydotplus.graph_from_dot_data(source)
        nx_graph = networkx.nx_pydot.from_pydot(dotplus)
        return nx_graph

    def add_label2word(self, label2word: Dict[str, str]):
        self.label2word = label2word

    def add_belief_argument(self, belief_argument):
        self.belief_argument = belief_argument

    def add_sentence_segments(self, segments):
        self.segments = segments


if __name__ == "__main__":

    x = joblib.load(
        "../../cache/masked_sentences_with_AMR_container_objects_train.joblib")
    embed()
    exit()

    # sentence = "she is the best because everybody loves her."
    # amr1 = AMR_Container(
    #     sentence=sentence
    # )

    # sentence = "she is the best because everybody loves her."
    # amr2 = AMR_Container(
    #     sentence=sentence
    # )
