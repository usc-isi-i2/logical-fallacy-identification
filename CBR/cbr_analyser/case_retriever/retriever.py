import argparse
import os
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm
import sys
from abc import abstractmethod
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from IPython import embed
from yaml import parse

# from cbr_analyser.case_retriever.gcn.gcn import (NODE_EMBEDDING_SIZE,
#  CBRetriever)

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))


class Retriever:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieve_similar_cases(self, case: str, num_cases: int):
        pass


# class GCN_Retriever(Retriever):
#     def __init__(self, config: Dict[str, Any]) -> None:
#         self.similarities_dict = dict()
#         simcse_model_paths = [
#             f"cache/gcn_similarities_{config['source_feature']}_{split}.joblib"
#             for split in ["train", "dev", "test"]
#         ]
#         for path in simcse_model_paths:
#             self.similarities_dict.update(joblib.load(path))

#     def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
#         sentences_and_similarities = self.similarities_dict[case.strip()].items(
#         )
#         sentences_and_similarities_sorted = sorted(
#             sentences_and_similarities, key=lambda x: x[1], reverse=True)

#         return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


# class Knn_Retriever(Retriever):
#     def __init__(self, model_path: str, sentences: List[str], num_cases: int) -> None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_path
#         ).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         cache_file_path = "cache/knn_retriever.joblib"
#         if os.path.exists(cache_file_path):
#             embeddings_means = joblib.load(cache_file_path)
#         else:
#             print('Calculating embeddings... for KNN retriever')
#             all_embeddings = []
#             for sentence in tqdm(sentences, leave=False):
#                 with torch.no_grad():
#                     inputs = tokenizer(sentence, return_tensors='pt',
#                                        truncation=True, padding='max_length', max_length=24)
#                     inputs = inputs.to(device)
#                     outputs = model(**inputs, output_hidden_states=True)
#                     last_layer_hidden_states = outputs.hidden_states[-1].detach().cpu().numpy(
#                     )
#                     all_embeddings.append(last_layer_hidden_states)
#             all_embeddings = np.array(all_embeddings)
#             embeddings_means = np.mean(
#                 all_embeddings, axis=2, keepdims=True).squeeze()
#             joblib.dump(embeddings_means, cache_file_path)

#         neigh = NearestNeighbors(n_neighbors=num_cases + 1)
#         neigh.fit(embeddings_means)
#         self.neigh = neigh
#         self.sentences = sentences
#         self.tokenizer = tokenizer
#         self.model = model

#     def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 case, return_tensors='pt', truncation=True, padding='max_length', max_length=24)
#             inputs = inputs.to(device)
#             outputs = self.model(**inputs, output_hidden_states=True)
#             last_layer_hidden_states = outputs.hidden_states[-1].detach(
#             ).cpu().numpy()
#             embedding_means = np.mean(
#                 last_layer_hidden_states, axis=1, keepdims=True).squeeze()

#         results = self.neigh.kneighbors([embedding_means])
#         indices = results[1][0]
#         try:
#             retriever_outputs = [(self.sentences[i], 1, train_df[train_df["text"].str.strip(
#             ) == self.sentences[i].strip()].label.tolist()) for i in indices[1:]]
#         except:
#             embed()
#             exit()
#         return retriever_outputs


class SimCSE_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        simcse_model_paths = [
            os.path.join(
                "cache", config["data_dir"].replace("/", "_"), f"simcse_similarities_{config['source_feature']}_{split}.joblib")
            for split in ["train", "dev", "test"]
        ]
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)

        return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


class AMR_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        simcse_model_paths = [
            os.path.join(
                "../../", "cache", config["data_dir"].replace("/", "_"), f"amr_similarities_{split}.joblib")
            for split in ["train", "dev", "test"]
        ]
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)

        return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


class Empathy_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        empathetic_model_paths = [
            os.path.join(
                "cache", config["data_dir"].replace("/", "_"), f"empathy_similarities_{config['source_feature']}_{split}.joblib")
            for split in ["train", "dev", "test"]
        ]
        for path in empathetic_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)

        return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


if __name__ == "__main__":
    pass
