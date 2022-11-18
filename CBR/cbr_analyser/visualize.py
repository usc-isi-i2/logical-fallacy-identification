import argparse
import os

import joblib
import matplotlib.pyplot as plt
import networkx as nx
from IPython import embed
import shutil
from tqdm import tqdm
import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../cbr_analyser/amr/"))


def get_networkx_graph_from_edge_lines(line: str) -> nx.Graph:
    edges_str = line[1:-1].split(')(')
    g = nx.Graph()
    for edge_str in edges_str:
        try:
            edge = edge_str.split(';')
            edge = [edge[0].replace(":", "").strip(), edge[1].replace(
                ":", "").strip(), edge[2].replace(":", "").strip()]
            if len(edge[0]) == 1 or len(edge[1]) == 1 or len(edge[2]) == 1:
                continue
            g.add_edge(edge[0], edge[2], label=edge[1])
        except Exception as e:
            embed()
            exit()
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualze the edges generated by ExplaGraph")
    parser.add_argument(
        '--input_file', help="The file to get the linearized graphs from")
    parser.add_argument(
        '--output_dir', help="The Folder to dump the visualizations in")

    parser.add_argument(
        '--input_type', help="type of the input", type=str)

    args = parser.parse_args()

    if args.input_type == "explagraph":

        with open(args.input_file, 'r') as f:
            data = f.read().splitlines()
        print(len(data))

        all_graphs = []
        for i, line in tqdm(enumerate(data), leave=False):
            if line != "":
                all_graphs.append(
                    (i + 1, get_networkx_graph_from_edge_lines(line=line))
                )
    elif args.input_type == "amr":
        sentences_with_amr_obj = joblib.load(args.input_file)
        all_graphs = []
        for index, obj in enumerate(sentences_with_amr_obj):
            all_graphs.append(
                (index + 1, obj[1].graph_nx, obj[1].sentence, obj[2]))

    shutil.rmtree(os.path.join(
                    args.output_dir, args.input_type))
    for index, graph, sentence, label in tqdm(all_graphs, leave=False):
        try:
            fig = plt.figure(figsize=(20, 20))
            fig.suptitle(f"{sentence} with label: {label}", fontsize=20)
            pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
            labels = {
                node[0]: node[1]['label']
                for node in graph.nodes(data=True)
            }
            nx.draw(G=graph, pos=pos, with_labels=True,
                    labels=labels, font_size=20)
            edge_labels = {(edge[0], edge[1]): edge[2]['label']
                           for edge in graph.edges(data=True)}
            nx.draw_networkx_edge_labels(
                G=graph, pos=pos, edge_labels=edge_labels, font_size = 10)
            if not os.path.exists(os.path.join(args.output_dir, args.input_type, label)):
                os.makedirs(os.path.join(
                    args.output_dir, args.input_type, label))

            plt.savefig(os.path.join(args.output_dir,
                        args.input_type, label, f"graph_{index}.png"))
            plt.close()
        except Exception as e:
            print(e)
            continue