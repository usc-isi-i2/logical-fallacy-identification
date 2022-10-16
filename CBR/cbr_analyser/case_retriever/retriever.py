import os
import sys
from abc import abstractmethod
from typing import Any, Dict, List

import joblib
import numpy as np
import torch

from cbr_analyser import consts
from cbr_analyser.case_retriever.gcn.gcn import (NODE_EMBEDDING_SIZE,
                                                 CBRetriever)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))


class Retriever:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieve_similar_cases(self, case: str, num_cases: int):
        pass


class GCN_Retriever(Retriever):
    def __init__(self, gcn_model_path: str, config: Dict[str, Any], train_input_file: str) -> None:
        self.model = CBRetriever(
            num_input_features=NODE_EMBEDDING_SIZE,
            num_output_features=len(consts.label2index),
            mid_layer_dropout=config["mid_layer_dropout"],
            mid_layer_embeddings=config["gcn_layers"]
        )

        self.model.load_state_dict(torch.load(gcn_model_path))

        self.model = self.model.to(device)
        self.train_objects = joblib.load(train_input_file)

    def retrieve_similar_cases(self, case: str, num_cases: int):
        pass


class SimCSE_Retriever(Retriever):
    def __init__(self, simcse_model_paths: List[str], config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, num_cases: int, threshold):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


class Empathetic_Retriever(Retriever):
    def __init__(self, empathetic_model_paths: str, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        for path in empathetic_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, num_cases: int, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


if __name__ == "__main__":
    pass
