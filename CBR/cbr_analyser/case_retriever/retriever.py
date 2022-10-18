import argparse
import os
import pickle
import sys
from abc import abstractmethod
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from IPython import embed
from yaml import parse

from cbr_analyser import consts
from cbr_analyser.case_retriever.gcn.gcn import (NODE_EMBEDDING_SIZE,
                                                 CBRetriever)

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))


class Retriever:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieve_similar_cases(self, case: str, num_cases: int):
        pass


class GCN_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        simcse_model_paths = [
            f"cache/gcn_similarities_{config['source_feature']}_{split}.joblib"
            for split in ["train", "dev", "test"]
        ]
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)
        max_score = sentences_and_similarities_sorted[0][1]
        return [(x[0], x[1] / max_score) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


class SimCSE_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        simcse_model_paths = [
            f"cache/simcse_similarities_{config['source_feature']}_{split}.joblib"
            for split in ["train", "dev", "test"]
        ]
        for path in simcse_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)
        max_score = sentences_and_similarities_sorted[0][1]
        return [(x[0], x[1] / max_score) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


class Empathy_Retriever(Retriever):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.similarities_dict = dict()
        empathetic_model_paths = [
            f"cache/empathy_similarities_{config['source_feature']}_{split}.joblib"
            for split in ["train", "dev", "test"]
        ]
        for path in empathetic_model_paths:
            self.similarities_dict.update(joblib.load(path))

    def retrieve_similar_cases(self, case: str, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)
        max_score = sentences_and_similarities_sorted[0][1]
        return [(x[0], x[1]/max_score) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


if __name__ == "__main__":
    pass
