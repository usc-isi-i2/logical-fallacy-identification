import networkx
import pydotplus
from IPython import embed
import networkx as nx
import amrlib
from amrlib.graph_processing.amr_plot import *
import graphviz    # sudo apt install graphviz; pip3 install graphviz


class AMR_Container:
    """
    its instances will contain the AMR representation of the sentences. 
    https://github.com/bjascob/amrlib
    """

    def __init__(self, sentence: str = None, graph_str: str = None) -> None:
        if sentence is not None:
            self.sentence = sentence
            self.graph_str = self.generate_amr()
        elif graph_str is not None:
            self.graph_str = graph_str
        else:
            raise NotImplementedError()
        self.graph_viz = self.generate_graphviz()
        self.graph_nx = self.generate_networkx()

    def generate_amr(self) -> str:
        """
        Create the AMR representation of the sentence

        Returns:
            str: AMR representation of the sentence
        """
        stog = amrlib.load_stog_model()
        graph = stog.parse_sents([self.sentence])[0]
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


if __name__ == "__main__":
    sentence = "she is the best because everybody loves her."
    amr = AMR(
        sentence=sentence
    )

    embed()
    exit()