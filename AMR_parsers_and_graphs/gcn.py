from operator import index
from torch_geometric.data import Data
import random
import re
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool, SAGEConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from consts import *
from IPython import embed
import joblib


class NormalNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(NormalNet, self).__init__()

        self.conv1 = SAGEConv(
            num_input_features,
            32
        )
        self.conv2 = SAGEConv(
            32,
            16
        )

        self.lin = nn.Linear(16, num_output_features)

    def forward(self, data, batch):
        try:
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)

            x = F.relu(x)
            x = self.conv2(x, edge_index)

            x = global_mean_pool(x, batch)

            x = F.dropout(x, training=self.training)
            x = self.lin(x)
        except Exception as e:
            print(e)
            embed()

        return x


class Logical_Fallacy_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.amr_graphs_with_sentences = joblib.load(
            PATH_TO_MASKED_SENTENCES_AMRS)
        self.edge_type2index = self.get_edge_mappings()
        self.label2index = self.get_label_mappings()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        return

    def get_edge_mappings(self):
        edge_type2index = {}
        all_edge_types = set()
        for obj in self.amr_graphs_with_sentences:
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

    def get_label_mappings(self):
        label2index = {}
        all_labels = set()
        for obj in self.amr_graphs_with_sentences:
            all_labels.add(obj[2])

        label2index = {
            label: index
            for index, label
            in enumerate(all_labels)
        }

        return label2index

    def get_node_mappings(self, g):
        node2index = {}
        for i, node in enumerate(g.nodes()):
            node2index[node] = i
        return node2index

    def process(self):
        # TODO add the embedding to the x property
        # TODO add the edge_types to the edge_type property
        data_list = []
        for obj in self.amr_graphs_with_sentences:
            g = obj[1].graph_nx
            node2index = self.get_node_mappings(g)
            new_g = nx.Graph()
            new_g.add_edges_from([
                (
                    node2index[edge[0]],
                    node2index[edge[1]],
                )
                for edge in g.edges(data=True)
            ])
            pyg_graph = from_networkx(new_g)
            pyg_graph.x = torch.Tensor([
                self.label2index[obj[2]]
            ] * pyg_graph.num_nodes).view(-1, 1)
            pyg_graph.y = torch.tensor([
                self.label2index[obj[2]]
            ])
            data_list.append(pyg_graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train(loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


if __name__ == "__main__":
    dataset = Logical_Fallacy_Dataset(root='.')
    dataset = dataset.shuffle()
    last_train_index = int(len(dataset) * 1)

    train_dataset = dataset[:last_train_index]
    # test_dataset = dataset[last_train_index:]

    print(f'Number of training graphs: {len(train_dataset)}')
    # print(f'Number of test graphs: {len(test_dataset)}')

    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NormalNet(
        num_input_features=1,
        num_output_features=len(dataset.label2index)
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=5e-4)

    for epoch in range(1, 171):
        train(train_data_loader)
        train_acc = test(train_data_loader)
        # test_acc = test(test_data_loader)
        # print(
        #     f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
