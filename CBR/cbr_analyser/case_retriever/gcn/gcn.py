import argparse
import os
import random
import re
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Dict, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython import embed
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (RGATConv, RGCNConv, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

import cbr_analyser.consts as consts
import wandb
from cbr_analyser.augmentations.embedding_extractor import get_bert_embeddings

NODE_EMBEDDING_SIZE = 768
BATCH_SIZE = 4
NUM_EPOCHS = 70
MID_LAYERS_DROPOUT = 0.1
GCN_LAYERS = "32,32,16"
LEARNING_RATE = 1e-4

# torch.manual_seed(77)
# random.seed(77)
# np.random.seed(77)


class CBRetriever(torch.nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int, mid_layer_dropout: float, mid_layer_embeddings: List[int], heads=4):
        super(CBRetriever, self).__init__()
        self.mid_layer_dropout = mid_layer_dropout

        self.g_layers = nn.ModuleList()
        self.g_layers.append(
            RGATConv(
                num_input_features,
                mid_layer_embeddings[0],
                num_relations=consts.num_edge_types
            )
        )

        for i in range(1, len(mid_layer_embeddings)):
            self.g_layers.append(
                RGATConv(
                    mid_layer_embeddings[i - 1],
                    mid_layer_embeddings[i],
                    num_relations=consts.num_edge_types
                )
            )

        self.lin = nn.Linear(2*mid_layer_embeddings[-1], num_output_features)

    def get_embedding(self, data, batch):
        try:
            x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
            edge_type = edge_type.long()
            # x = F.dropout(x, p=0.1, training=self.training)

            for i in range(len(self.g_layers)):
                x = self.g_layers[i](x, edge_index, edge_type)
                if i != len(self.g_layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.mid_layer_dropout,
                                  training=self.training)

            x = torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

            return x.detach().cpu()

        except Exception as e:
            print(e)
            embed()
            exit()

        return x

    def forward(self, data, batch):
        try:
            x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
            edge_type = edge_type.long()
            # x = F.dropout(x, p=0.1, training=self.training)

            for i in range(len(self.g_layers)):
                x = self.g_layers[i](x, edge_index, edge_type)
                if i != len(self.g_layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.mid_layer_dropout,
                                  training=self.training)

            x = torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.lin(x)

        except Exception as e:
            print(e)
            embed()
            exit()

        return x


class Logical_Fallacy_Dataset(InMemoryDataset):
    def __init__(self, path_to_dataset, g_type: str, fit=False, **kwargs):
        self.path_to_dataset = path_to_dataset
        self.g_type = g_type
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

        self.edge_mapping = {edge: i for i,
                             edge in enumerate(self.all_edge_types)}
        super().__init__(root='.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return len(self.all_edge_types)

    @ property
    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.path_to_dataset))[0]}_{self.g_type}.pt']

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
                base_sentence = obj[1].sentence
                g = obj[1].graph_nx
                label2word = obj[1].label2word
                node2index = self.get_node_mappings(g)
                index2embeddings = self.get_node_embeddings(g, label2word)
                if self.g_type == "directed":
                    new_g = nx.DiGraph()
                else:
                    new_g = nx.Graph()
                all_edges = {
                    (
                        node2index[edge[0]],
                        node2index[edge[1]],
                    ): self.edge_mapping[edge[2]['label']]
                    for edge in g.edges(data=True)
                }
                new_g.add_edges_from(list(all_edges.keys()))
                nx.set_node_attributes(new_g, index2embeddings, name='x')
                nx.set_edge_attributes(new_g, all_edges, 'edge_type')
                pyg_graph = from_networkx(new_g)

                # edge_attrs = np.array(pyg_graph.edge_attr).reshape(-1, 1)
                # pyg_graph.edge_attr = torch.from_numpy(
                #     self.ohe.transform(edge_attrs).A
                # )

                pyg_graph.y = torch.tensor([
                    consts.label2index[obj[2]]
                ])

                pyg_graph.base_sentence = base_sentence

                data_list.append(pyg_graph)
            except Exception as e:
                print(e)
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train_epoch(model, loader, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test_on_loader(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_true_labels = []
    all_sentences = []
    all_confs = []

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        all_sentences.extend(data.base_sentence)
        data = data.to(device)
        out = model(data, data.batch)
        probs = torch.nn.functional.softmax(out, dim=1)
        confs = torch.max(probs, dim=1).values
        all_confs.extend(confs.tolist())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        all_predictions.extend(pred.tolist())
        all_true_labels.extend(data.y.tolist())
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return {
        'acc': correct / len(loader.dataset),
        'predictions': all_predictions,
        'all_sentences': all_sentences,
        'true_labels': all_true_labels,
        'confidence': all_confs
    }


def evaluate_on_loaders(model, train_data_loader, dev_data_loader, test_data_loader):
    train_results = test_on_loader(
        model, train_data_loader)
    dev_results = test_on_loader(
        model, dev_data_loader)
    test_results = test_on_loader(
        model, test_data_loader)

    all_train_predictions = [consts.index2label[pred]
                             for pred in train_results['predictions']]
    all_dev_predictions = [consts.index2label[pred]
                           for pred in dev_results['predictions']]
    all_test_predictions = [consts.index2label[pred]
                            for pred in test_results['predictions']]
    all_train_true_labels = [consts.index2label[true_label]
                             for true_label in train_results['true_labels']]
    all_dev_true_labels = [consts.index2label[true_label]
                           for true_label in dev_results['true_labels']]
    all_test_true_labels = [consts.index2label[true_label]
                            for true_label in test_results['true_labels']]

    print('Train split results')
    print(classification_report(
        y_pred=all_train_predictions,
        y_true=all_train_true_labels
    ))

    print('Dev split results')
    print(classification_report(
        y_pred=all_dev_predictions,
        y_true=all_dev_true_labels
    ))

    print('Test split results')
    print(classification_report(
        y_pred=all_test_predictions,
        y_true=all_test_true_labels
    ))


def get_similarities(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["train_input_file"],
        g_type=config["g_type"],
        fit=True
    )

    dev_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["dev_input_file"],
        g_type=config["g_type"],
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    test_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["test_input_file"],
        g_type=config["g_type"],
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    batch_size = config["batch_size"]

    train_data_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False)
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=config["batch_size"], shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CBRetriever(
        num_input_features=NODE_EMBEDDING_SIZE,
        num_output_features=len(consts.label2index),
        mid_layer_dropout=config["mid_layer_dropout"],
        mid_layer_embeddings=config["gcn_layers"],
        heads=4
    )
    model.load_state_dict(torch.load(config["gcn_model_path"]))
    model = model.to(device)

    model.eval()

    model = model.to(device)
    for source_data, data_name in zip([train_data_loader, dev_data_loader, test_data_loader], ['train', 'dev', 'test']):
        similarities = np.zeros([len(source_data.dataset),
                                len(train_data_loader.dataset)])
        all_sentences = [data.base_sentence.strip()
                         for data in source_data.dataset]
        train_sentences = [
            data.base_sentence.strip() for data in train_data_loader.dataset]
        for source_index, source_batch in enumerate(source_data):
            for train_index, train_batch in enumerate(train_data_loader):

                source_batch = source_batch.to(device)
                train_batch = train_batch.to(device)

                source_embedding = model.get_embedding(
                    source_batch, source_batch.batch)
                train_embedding = model.get_embedding(
                    train_batch, train_batch.batch)

                similarities[
                    source_index * batch_size:source_index * batch_size + len(source_batch),
                    train_index * batch_size:train_index * batch_size + len(train_batch)
                ] = pairwise_cosine_similarity(
                    source_embedding,
                    train_embedding
                ).numpy()

        similarities_dict = dict()
        for sentence, row in zip(all_sentences, similarities):
            similarities_dict[sentence] = dict(
                zip(train_sentences, row.tolist()))
        output_file = f"cache/gcn_similarities_{config['source_feature']}_{data_name}.joblib"
        joblib.dump(similarities_dict, output_file)


def do_train_process(model, train_data_loader, dev_data_loader, test_data_loader, learning_rate, num_epochs, gcn_model_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    best_f1_score = -np.inf
    for epoch in range(1, num_epochs):
        train_epoch(model, train_data_loader, optimizer, criterion)

        train_results = test_on_loader(
            model, train_data_loader)
        dev_results = test_on_loader(
            model, dev_data_loader)

        score = f1_score(
            y_true=dev_results['true_labels'],
            y_pred=dev_results['predictions'],
            average="weighted"
        )

        wandb.log({
            "Train Accuracy": train_results['acc'],
            "Dev Accuracy": dev_results['acc'],
            'dev_f1_score': score
        }, step=epoch)

        if score > best_f1_score:
            print(score)
            print('saving the model')
            torch.save(model.state_dict(), gcn_model_path)
            best_f1_score = score

        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_results['acc']:.4f}, Dev Acc: {dev_results['acc']:.4f}")
        if epoch == num_epochs - 1:
            evaluate_on_loaders(model, train_data_loader,
                                dev_data_loader, test_data_loader)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="Logical Fallacy Detection GCN",
        entity='zhpinkman',
        config={
            **config,
            "model": "RGATConv"
        }
    )

    train_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["train_input_file"],
        g_type=config["g_type"],
        fit=True
    )

    dev_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["dev_input_file"],
        g_type=config["g_type"],
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    test_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=config["test_input_file"],
        g_type=config["g_type"],
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True)
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=config["batch_size"], shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CBRetriever(
        num_input_features=NODE_EMBEDDING_SIZE,
        num_output_features=len(consts.label2index),
        mid_layer_dropout=config["mid_layer_dropout"],
        mid_layer_embeddings=config["gcn_layers"],
        heads=4
    )

    model = model.to(device)

    do_train_process(
        model=model,
        train_data_loader=train_data_loader,
        dev_data_loader=dev_data_loader,
        test_data_loader=test_data_loader,
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        gcn_model_path=config["gcn_model_path"]
    )


def train_with_wandb(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        train(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training and predicting the logical Fallacy types using Graph Neural Networks")
    parser.add_argument('--task', choices=['train', 'predict', 'hptuning'],
                        help="The task you want the model to accomplish")
    parser.add_argument(
        '--gcn_model_path', help="The path from which we can find the pre-trained model")

    parser.add_argument('--train_input_file',
                        help="The path to the train dataset")

    parser.add_argument('--dev_input_file',
                        help="The path to the dev dataset")

    parser.add_argument('--test_input_file',
                        help="The path to the test data")

    parser.add_argument(
        '--predictions_path', type=str
    )
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--g_type', help="The type of graph you want to use",
                        choices=['directed', 'undirected'])
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--mid_layers_dropout', type=float,
                        default=MID_LAYERS_DROPOUT)
    parser.add_argument('--gcn_layers',
                        type=str, default=GCN_LAYERS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    if args.task == "hptuning":

        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'dev_f1_score',
            'goal': 'maximize'
        }

        sweep_config['metric'] = metric

        parameters_dict = {
            'batch_size': {
                'values': [4]
            },
            'num_epochs': {
                "value": 400
            },
            'mid_layer_dropout': {
                "values": [0.5, 0.6, 0.7, 0.8]
            },
            'gcn_layers': {
                'values':
                    [
                        [32, 32, 32],
                        [32, 32, 16],
                        [16, 16, 16],
                        [64, 32, 32]
                    ]

            },
            'learning_rate': {
                'values': [1e-4, 1e-5, 5e-5]
            }
        }

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(
            sweep_config, project="Logical Fallacy Detection GCN Hyper parameter tuning V2")

        wandb.agent(sweep_id, train_with_wandb)
        exit()

    wandb.init(
        project="Logical Fallacy Detection GCN",
        entity='zhpinkman',
        config={
            "model": "GATv2Conv"
        }
    )

    train_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.train_input_file,
        g_type=args.g_type,
        fit=True
    )

    dev_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.dev_input_file,
        g_type=args.g_type,
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    test_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.test_input_file,
        g_type=args.g_type,
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CBRetriever(
        num_input_features=NODE_EMBEDDING_SIZE,
        num_output_features=len(consts.label2index),
        mid_layer_dropout=args.mid_layers_dropout,
        mid_layer_embeddings=[int(x)
                              for x in args.gcn_layers.split(',')],
        heads=4
    )
    if args.task == "predict":
        model.load_state_dict(torch.load(args.gcn_model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.task == "train":
        do_train_process(
            model=model,
            train_data_loader=train_data_loader,
            dev_data_loader=dev_data_loader,
            test_data_loader=test_data_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            gcn_model_path=args.gcn_model_path
        )
        exit()

    elif args.task == "predict":
        evaluate_on_loaders(model, train_data_loader,
                            dev_data_loader, test_data_loader)
        test_results = test_on_loader(model=model, loader=test_data_loader)
        test_predictions = [consts.index2label[pred]
                            for pred in test_results['predictions']]
        test_true_labels = [consts.index2label[true_label]
                            for true_label in test_results['true_labels']]

        df = pd.DataFrame({
            "sentence": test_results['all_sentences'],
            "prediction": test_predictions,
            "true_label": test_true_labels
        })
        df.to_csv(os.path.join(args.predictions_path,
                  "gcn_predictions_test.csv"), index=False)
