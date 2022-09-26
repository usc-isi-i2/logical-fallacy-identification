import wandb
from eval import get_random_predictions_for_pyg_metrics
from embeddings import get_bert_embeddings
import numpy as np
import os
import re
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool, GATv2Conv, TransformerConv
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.utils.convert import from_networkx
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset, Data
from consts import *
from IPython import embed
import joblib
from tqdm import tqdm
import argparse

BATCH_SIZE = 16
NUM_EPOCHS = 40
MID_LAYER_DROPOUT = 0.5
LAYERS_EMBEDDINGS = [128]
LEARNING_RATE = 1e-4

label2index = {
    'faulty generalization': 0,
    'false dilemma': 1,
    'false causality': 2,
    'appeal to emotion': 3,
    'ad hominem': 4,
    'fallacy of logic': 5,
    'intentional': 6,
    'equivocation': 7,
    'fallacy of extension': 8,
    'fallacy of credibility': 9,
    'ad populum': 10,
    'circular reasoning': 11,
    'fallacy of relevance': 12
}


parser = argparse.ArgumentParser(
    description="Training and predicting the logical Fallacy types using Graph Neural Networks")
parser.add_argument('--task', choices=['train', 'predict'],
                    help="The task you want the model to accomplish")
parser.add_argument(
    '--model_path', help="The path from which we can find the pre-trained model")

parser.add_argument('--all_data',
                    help="The path to the whole dataset")

parser.add_argument('--train_input_file',
                    help="The path to the train dataset")

parser.add_argument('--dev_input_file',
                    help="The path to the dev dataset")

parser.add_argument('--test_input_file',
                    help="The path to the test data")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class CBRetriever(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, mid_layer_dropout, mid_layer_embeddings, edge_dim, heads=8):
        super(CBRetriever, self).__init__()
        self.mid_layer_dropout = mid_layer_dropout

        self.conv1 = GATv2Conv(
            num_input_features,
            mid_layer_embeddings[0],
            heads=1
        )
        # self.conv2 = TransformerConv(
        #     heads * mid_layer_embeddings[0],
        #     mid_layer_embeddings[1],
        #     dropout=0.1,
        #     edge_dim=1,
        #     heads=heads // 2
        # )
        # self.conv3 = TransformerConv(
        #     heads // 2 * mid_layer_embeddings[1],
        #     mid_layer_embeddings[2],
        #     edge_dim=1,
        # )

        self.lin = nn.Linear(mid_layer_embeddings[-1], num_output_features)

    def forward(self, data, batch):
        try:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr = edge_attr.float()

            x = self.conv1(x, edge_index)

            # x = F.relu(x)
            # x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            # x = self.conv2(x, edge_index, edge_attr)

            # x = F.relu(x)
            # x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            # x = self.conv3(x, edge_index, edge_attr)

            x = global_mean_pool(x, batch)

            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.lin(x)

        except Exception as e:
            print(e)
            embed()
            exit()

        return x


class Logical_Fallacy_Dataset(InMemoryDataset):
    def __init__(self, path_to_dataset, fit=False, **kwargs):
        self.path_to_dataset = path_to_dataset
        self.amr_graphs_with_sentences = joblib.load(
            path_to_dataset)
        if fit:
            self.all_edge_types = self.get_all_edge_types()
            self.ohe = OneHotEncoder()
            self.ohe.fit(
                np.array(list(self.all_edge_types)).reshape(-1, 1)
            )
        else:
            self.all_edge_types = kwargs["all_edge_types"]
            self.ohe = kwargs["ohe"]

        super().__init__(root='.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.path_to_dataset))[0]}.pt']

    # @property
    # def num_relations(self) -> int:
    #     return int(self.data.edge_type.max()) + 1

    def _download(self):
        return

    def get_all_edge_types(self):
        all_edge_types = set()
        for obj in self.amr_graphs_with_sentences:
            g = obj[1].graph_nx
            for edge in g.edges(data=True):
                arg = edge[2]['label']
                all_edge_types.add(arg)

        return all_edge_types

    def get_node_embeddings(self, g, label2word):
        index2embeddings = {}
        for i, node in enumerate(g.nodes()):
            index2embeddings[i] = get_bert_embeddings(
                label2word[node]).tolist()
        return index2embeddings

    def get_node_mappings(self, g):
        node2index = {}
        for i, node in enumerate(g.nodes()):
            node2index[node] = i
        return node2index

    def process(self):
        data_list = []
        for obj in tqdm(self.amr_graphs_with_sentences, leave=False):
            try:
                g = obj[1].graph_nx
                label2word = obj[1].label2word
                node2index = self.get_node_mappings(g)
                index2embeddings = self.get_node_embeddings(g, label2word)
                new_g = nx.Graph()
                all_edges = {
                    (
                        node2index[edge[0]],
                        node2index[edge[1]],
                    ): edge[2]['label']
                    for edge in g.edges(data=True)
                }
                new_g.add_edges_from(list(all_edges.keys()))
                nx.set_node_attributes(new_g, index2embeddings, name='x')
                nx.set_edge_attributes(new_g, all_edges, 'edge_attr')
                pyg_graph = from_networkx(new_g)

                edge_attrs = np.array(pyg_graph.edge_attr).reshape(-1, 1)
                pyg_graph.edge_attr = torch.from_numpy(
                    self.ohe.transform(edge_attrs).A
                )

                pyg_graph.y = torch.tensor([
                    label2index[obj[2]]
                ])

                data_list.append(pyg_graph)
            except Exception as e:
                print(e)
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train(model, loader, optimizer):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, loader):
    model.eval()

    all_predictions = []
    all_true_labels = []

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        all_predictions.extend(pred.tolist())
        all_true_labels.extend(data.y.tolist())
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset), all_predictions, all_true_labels


if __name__ == "__main__":
    print('train data')
    train_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.train_input_file,
        fit=True
    )

    print('dev data')
    dev_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.dev_input_file,
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    print('test data')
    test_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.test_input_file,
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    # for i in range(299):
    #     a = dataset[1849 + i]
    #     b = dev_dataset[i]

    #     if (not torch.all(torch.eq(a.x, b.x))) or (not torch.all(torch.eq(a.edge_index, b.edge_index))) or (not torch.all(torch.eq(a.y, b.y))):
    #         embed()

    # for i in range(299):
    #     a = dataset[2149 + i]
    #     b = test_dataset[i]

    #     if (not torch.all(torch.eq(a.x, b.x))) or (not torch.all(torch.eq(a.edge_index, b.edge_index))) or (not torch.all(torch.eq(a.y, b.y))):
    #         embed()
    # exit()

    # dev_dataset = Logical_Fallacy_Dataset(
    #     path_to_dataset=args.dev_input_file,
    #     ohe=train_dataset.ohe,
    #     all_edge_types=train_dataset.all_edge_types,
    # )
    # test_dataset = Logical_Fallacy_Dataset(
    #     path_to_dataset=args.test_input_file,
    #     ohe=train_dataset.ohe,
    #     all_edge_types=train_dataset.all_edge_types,
    # )

    # train_dataset = dataset[:1849]
    # dev_dataset = dataset[1849:2149]
    # test_dataset = dataset[2149:]

    # wandb.init(
    #     project="Logical Fallacy Detection GCN",
    #     entity='zhpinkman',
    #     config={
    #         "learning_rate": LEARNING_RATE,
    #         "epochs": NUM_EPOCHS,
    #         "mid_layer_dropout": MID_LAYER_DROPOUT,
    #         "layers_embeddings": LAYERS_EMBEDDINGS,
    #         "batch_size": BATCH_SIZE,
    #         "edge_attr": True,
    #         "model": "GATv2Conv"
    #     }
    # )

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f"Number of dev graphs: {len(dev_dataset)}")
    print(f'Number of test graphs: {len(test_dataset)}')

    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CBRetriever(
        num_input_features=train_dataset.num_features,
        num_output_features=len(label2index),
        mid_layer_dropout=MID_LAYER_DROPOUT,
        mid_layer_embeddings=LAYERS_EMBEDDINGS,
        edge_dim=train_dataset.ohe.get_feature_names().shape[0]
    )
    if args.task == "predict":
        model.load_state_dict(torch.load(args.model_path))

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS):
        train(model, train_data_loader, optimizer)

        train_acc, all_train_predictions, all_train_true_labels = test(
            model, train_data_loader)
        dev_acc, _, _ = test(
            model, dev_data_loader)

        # wandb.log({
        #     "Train Accuracy": train_acc,
        #     "Dev Accuracy": dev_acc
        # }, step=epoch)

        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {dev_acc:.4f}')
        if epoch == NUM_EPOCHS - 1:
            train_acc, all_train_predictions, all_train_true_labels = test(
                model, train_data_loader)
            dev_acc, all_dev_predictions, all_dev_true_labels = test(
                model, dev_data_loader)
            test_acc, all_test_predictions, all_test_true_labels = test(
                model, test_data_loader)

            index2label = {v: k for k, v in label2index.items()}

            all_train_predictions = [index2label[pred]
                                     for pred in all_train_predictions]
            all_dev_predictions = [index2label[pred]
                                   for pred in all_dev_predictions]
            all_test_predictions = [index2label[pred]
                                    for pred in all_test_predictions]
            all_train_true_labels = [index2label[true_label]
                                     for true_label in all_train_true_labels]
            all_dev_true_labels = [index2label[true_label]
                                   for true_label in all_dev_true_labels]
            all_test_true_labels = [index2label[true_label]
                                    for true_label in all_test_true_labels]

            print(classification_report(
                y_pred=all_test_predictions,
                y_true=all_test_true_labels
            ))

            print(get_random_predictions_for_pyg_metrics(
                dataset=train_dataset,
                all_test_true_labels=all_train_true_labels,
                size=100,
                label2index = label2index
            ))
    if args.task == "train":
        torch.save(model.state_dict(), args.model_path)
