from eval import get_random_predictions_for_pyg_metrics
from embeddings import get_bert_embeddings
import re
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool, SAGEConv
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.utils.convert import from_networkx
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from consts import *
from IPython import embed
import joblib
from tqdm import tqdm


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

    def get_node_embeddings(self, g):
        label2embeddings = {}

        for i, node in enumerate(g.nodes(data=True)):
            try:
                label = node[1]['label']
                if label == '"-"':
                    label = '"negative"'
                if label == '"+"':
                    label = '"positive"'
                if not re.match(r'".*"', label):
                    label = f'"{label}"'

                if '/' in label and '-' in label:
                    pattern = r'"[a-zA-Z0-9]+/([a-zA-Z-]+)(-\d*)?"'
                    word = re.findall(pattern, label)[0][0]
                    word = re.sub('-', ' ', word)
                elif '/' in label and '-' not in label:
                    pattern = r'"[a-zA-Z0-9]+/([\w]+)"'
                    word = re.findall(pattern, label)[0]

                else:
                    word = re.findall(r'"(.*)"', label)[0]

                if word == " ":
                    print(label)
                    embed()
            except Exception as e:
                print(e)
                print(label)
                word = label

            label2embeddings[i] = get_bert_embeddings(word).tolist()
        return label2embeddings

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
        for obj in tqdm(self.amr_graphs_with_sentences, leave=False):
            g = obj[1].graph_nx
            node2index = self.get_node_mappings(g)
            label2embeddings = self.get_node_embeddings(g)
            new_g = nx.Graph()
            new_g.add_edges_from([
                (
                    node2index[edge[0]],
                    node2index[edge[1]],
                )
                for edge in g.edges(data=True)
            ])
            nx.set_node_attributes(new_g, label2embeddings, name='x')
            pyg_graph = from_networkx(new_g)
            pyg_graph.y = torch.tensor([
                self.label2index[obj[2]]
            ])
            data_list.append(pyg_graph)
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
        out = model(data, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        all_predictions.extend(pred.tolist())
        all_true_labels.extend(data.y.tolist())
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset), all_predictions, all_true_labels


if __name__ == "__main__":
    dataset = Logical_Fallacy_Dataset(root='.')
    dataset = dataset.shuffle()
    last_train_index = int(len(dataset) * .8)

    train_dataset = dataset[:last_train_index]
    test_dataset = dataset[last_train_index:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NormalNet(
        num_input_features=dataset.num_features,
        num_output_features=len(dataset.label2index)
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=5e-4)

    num_epochs = 200

    for epoch in range(1, num_epochs):
        train(model, train_data_loader, optimizer)
        train_acc, all_train_predictions, all_train_true_labels = test(
            model, train_data_loader)
        test_acc, all_test_predictions, all_test_true_labels = test(
            model, test_data_loader)
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epoch == num_epochs - 1:
            label2index = dataset.label2index
            index2label = {v: k for k, v in label2index.items()}

            all_test_predictions = [index2label[pred]
                                    for pred in all_test_predictions]
            all_test_true_labels = [index2label[true_label]
                                    for true_label in all_test_true_labels]

            print(get_random_predictions_for_pyg_metrics(
                dataset=dataset,
                all_test_true_labels=all_test_true_labels,
                size=100
            ))

            print(classification_report(
                y_pred=all_test_predictions,
                y_true=all_test_true_labels
            ))

        # print(
        #     f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
