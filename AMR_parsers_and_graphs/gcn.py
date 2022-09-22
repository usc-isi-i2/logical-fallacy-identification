import wandb
from eval import get_random_predictions_for_pyg_metrics
from embeddings import get_bert_embeddings
import re
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool, GATv2Conv, TransformerConv
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

BATCH_SIZE = 16
NUM_EPOCHS = 80
MID_LAYER_DROPOUT = 0.2
LAYERS_EMBEDDINGS = [128, 64, 32]
LEARNING_RATE = 1e-4

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
print(device)


class NormalNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, mid_layer_dropout, mid_layer_embeddings, heads=8):
        super(NormalNet, self).__init__()
        self.mid_layer_dropout = mid_layer_dropout

        self.conv1 = TransformerConv(
            num_input_features,
            mid_layer_embeddings[0],
            dropout=0.1,
            edge_dim=1,
            heads=heads
        )
        self.conv2 = TransformerConv(
            heads * mid_layer_embeddings[0],
            mid_layer_embeddings[1],
            dropout=0.1,
            edge_dim=1,
            heads=heads // 2
        )
        self.conv3 = TransformerConv(
            heads // 2 * mid_layer_embeddings[1],
            mid_layer_embeddings[2],
            edge_dim=1,
        )

        self.lin = nn.Linear(mid_layer_embeddings[2], num_output_features)

    def forward(self, data, batch):
        try:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr = edge_attr.float()

            x = self.conv1(x, edge_index, edge_attr)

            x = F.relu(x)
            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.conv2(x, edge_index, edge_attr)

            x = F.relu(x)
            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)

            x = self.conv3(x, edge_index, edge_attr)

            x = global_mean_pool(x, batch)

            x = F.dropout(x, p=self.mid_layer_dropout, training=self.training)
            x = self.lin(x)
        except Exception as e:
            print(e)
            embed()
            exit()

        return x


class Logical_Fallacy_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.amr_graphs_with_sentences = joblib.load(
            PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD)
        self.edge_type2index = self.get_edge_mappings()
        self.label2index = self.get_label_mappings()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    # @property
    # def num_relations(self) -> int:
    #     return int(self.data.edge_type.max()) + 1

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

    def get_node_embeddings(self, g, label2word):
        index2embeddings = {}
        for i, node in enumerate(g.nodes()):
            index2embeddings[i] = get_bert_embeddings(
                label2word[node]).tolist()
        return index2embeddings

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
            label2word = obj[1].label2word
            node2index = self.get_node_mappings(g)
            index2embeddings = self.get_node_embeddings(g, label2word)
            new_g = nx.Graph()
            all_edges = {
                (
                    node2index[edge[0]],
                    node2index[edge[1]],
                ): self.edge_type2index[edge[2]['label']]
                for edge in g.edges(data=True)
            }

            new_g.add_edges_from(list(all_edges.keys()))
            nx.set_node_attributes(new_g, index2embeddings, name='x')
            nx.set_edge_attributes(new_g, all_edges, 'edge_attr')
            pyg_graph = from_networkx(new_g)
            pyg_graph.edge_attr = pyg_graph.edge_attr.reshape(-1, 1)
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
    dataset = Logical_Fallacy_Dataset(root='.')
    dataset = dataset.shuffle()

    last_train_index = int(len(dataset) * .8)

    wandb.init(
        project="Logical Fallacy Detection GCN",
        entity='zhpinkman',
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "mid_layer_dropout": MID_LAYER_DROPOUT,
            "layers_embeddings": LAYERS_EMBEDDINGS,
            "batch_size": BATCH_SIZE,
            "edge_attr": True,
            "model": "TransformerConv"
        }
    )

    train_dataset = dataset[:last_train_index]
    test_dataset = dataset[last_train_index:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NormalNet(
        num_input_features=dataset.num_features,
        num_output_features=len(dataset.label2index),
        mid_layer_dropout=MID_LAYER_DROPOUT,
        mid_layer_embeddings=LAYERS_EMBEDDINGS
    )

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS):
        train(model, train_data_loader, optimizer)
        train_acc, all_train_predictions, all_train_true_labels = test(
            model, train_data_loader)
        test_acc, all_test_predictions, all_test_true_labels = test(
            model, test_data_loader)

        wandb.log({
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc
        }, step=epoch)

        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epoch == NUM_EPOCHS - 1:
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
