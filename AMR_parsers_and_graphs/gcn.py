import wandb
from sklearn.metrics import f1_score
from embeddings import get_bert_embeddings
import numpy as np
import os
import re
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool, GATv2Conv
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.utils.convert import from_networkx
import torch
import random
import networkx as nx
from torch_geometric.data import InMemoryDataset
from consts import *
from IPython import embed
import joblib
from tqdm import tqdm
import argparse

NODE_EMBEDDING_SIZE = 768
BATCH_SIZE = 4
NUM_EPOCHS = 70
MID_LAYERS_DROPOUT = 0.1
MID_LAYERS_EMBEDDINGS = "32&32&16"
LEARNING_RATE = 1e-4

torch.manual_seed(77)
random.seed(77)
np.random.seed(77)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class CBRetriever(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, mid_layer_dropout, mid_layer_embeddings, heads=4):
        super(CBRetriever, self).__init__()
        self.mid_layer_dropout = mid_layer_dropout

        self.conv1 = GATv2Conv(
            num_input_features,
            mid_layer_embeddings[0],
            dropout=0.1,
            heads=heads
        )

        self.conv2 = GATv2Conv(
            heads * mid_layer_embeddings[0],
            mid_layer_embeddings[1],
            dropout=0.1,
            heads=heads // 2
        )

        self.conv3 = GATv2Conv(
            heads // 2 * mid_layer_embeddings[1],
            mid_layer_embeddings[2],
            dropout=0.1,
            heads=1
        )

        self.lin = nn.Linear(2*mid_layer_embeddings[-1], num_output_features)

    def forward(self, data, batch):
        try:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr = edge_attr.float()

            x = self.conv1(x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.conv2(x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.conv3(x, edge_index)

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
                pyg_graph.base_sentence = base_sentence

                data_list.append(pyg_graph)
            except Exception as e:
                print(e)
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train_epoch(model, loader, optimizer, criterion):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test_on_loader(model, loader):
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

    index2label = {v: k for k, v in label2index.items()}

    all_train_predictions = [index2label[pred]
                             for pred in train_results['predictions']]
    all_dev_predictions = [index2label[pred]
                           for pred in dev_results['predictions']]
    all_test_predictions = [index2label[pred]
                            for pred in test_results['predictions']]
    all_train_true_labels = [index2label[true_label]
                             for true_label in train_results['true_labels']]
    all_dev_true_labels = [index2label[true_label]
                           for true_label in dev_results['true_labels']]
    all_test_true_labels = [index2label[true_label]
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


def do_train_process(model, train_data_loader, dev_data_loader, test_data_loader, learning_rate, num_epochs):
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
            torch.save(model.state_dict(), args.model_path)
            best_f1_score = score

        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_results['acc']:.4f}, Dev Acc: {dev_results['acc']:.4f}")
        if epoch == num_epochs - 1:
            evaluate_on_loaders(model, train_data_loader,
                                dev_data_loader, test_data_loader)


def train_with_wandb(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        model = CBRetriever(
            num_input_features=NODE_EMBEDDING_SIZE,
            num_output_features=len(label2index),
            mid_layer_dropout=config.mid_layer_dropout,
            mid_layer_embeddings=config.layers_embeddings,
            heads=4
        )

        model = model.to(device)

        train_dataset = Logical_Fallacy_Dataset(
            path_to_dataset=args.train_input_file,
            fit=True
        )

        dev_dataset = Logical_Fallacy_Dataset(
            path_to_dataset=args.dev_input_file,
            fit=False,
            all_edge_types=train_dataset.all_edge_types,
            ohe=train_dataset.ohe
        )

        test_dataset = Logical_Fallacy_Dataset(
            path_to_dataset=args.test_input_file,
            fit=False,
            all_edge_types=train_dataset.all_edge_types,
            ohe=train_dataset.ohe
        )

        train_data_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True)
        dev_data_loader = DataLoader(
            dev_dataset, batch_size=config.batch_size, shuffle=False)
        test_data_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False)

        do_train_process(
            model=model,
            train_data_loader=train_data_loader,
            dev_data_loader=dev_data_loader,
            test_data_loader=test_data_loader,
            learning_rate=config.learning_rate,
            num_epochs=int(config.num_epochs),
        )


def do_predict_process(model, loader):
    index2label = {v: k for k, v in label2index.items()}
    test_results = test_on_loader(model, loader)
    all_sentences = test_results['all_sentences']
    predictions = test_results['predictions']
    predictions = [index2label[pred] for pred in predictions]
    confidence = test_results['confidence']
    return predictions, confidence, all_sentences


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training and predicting the logical Fallacy types using Graph Neural Networks")
    parser.add_argument('--task', choices=['train', 'predict', 'hptuning'],
                        help="The task you want the model to accomplish")
    parser.add_argument(
        '--model_path', help="The path from which we can find the pre-trained model")

    parser.add_argument('--train_input_file',
                        help="The path to the train dataset")

    parser.add_argument('--dev_input_file',
                        help="The path to the dev dataset")

    parser.add_argument('--test_input_file',
                        help="The path to the test data")

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--mid_layers_dropout', type=float,
                        default=MID_LAYERS_DROPOUT)
    parser.add_argument('--mid_layers_embeddings',
                        type=str, default=MID_LAYERS_EMBEDDINGS)
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
            'layers_embeddings': {
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

    model = CBRetriever(
        num_input_features=NODE_EMBEDDING_SIZE,
        num_output_features=len(label2index),
        mid_layer_dropout=args.mid_layers_dropout,
        mid_layer_embeddings=[int(x)
                              for x in args.mid_layers_embeddings.split('&')],
        heads=4
    )
    if args.task == "predict":
        model.load_state_dict(torch.load(args.model_path))

    model = model.to(device)

    train_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.train_input_file,
        fit=True
    )

    dev_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.dev_input_file,
        fit=False,
        all_edge_types=train_dataset.all_edge_types,
        ohe=train_dataset.ohe
    )

    test_dataset = Logical_Fallacy_Dataset(
        path_to_dataset=args.test_input_file,
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

    if args.task == "train":
        do_train_process(
            model=model,
            train_data_loader=train_data_loader,
            dev_data_loader=dev_data_loader,
            test_data_loader=test_data_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        exit()

    elif args.task == "predict":
        evaluate_on_loaders(model, train_data_loader,
                            dev_data_loader, test_data_loader)
        _, _, _ = do_predict_process(
            model=model,
            loader=test_data_loader
        )
        exit()
