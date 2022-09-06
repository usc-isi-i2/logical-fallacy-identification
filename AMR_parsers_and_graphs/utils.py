from consts import PATH_TO_MASKED_SENTENCES_AMRS, PATH_TO_MASKED_SENTENCES_AMRS, PATH_TO_MOST_SIMILAR_GRAPHS, PATH_TO_STATISTICS
from amr_container import AMR_Container
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from typing import Any, List
import networkx as nx
import numpy as np
import joblib
import json
import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from IPython import embed

import warnings
warnings.filterwarnings("ignore")

from amr_container import AMR_Container
from consts import PATH_TO_MASKED_SENTENCES_AMRS, PATH_TO_MASKED_SENTENCES_AMRS, PATH_TO_MOST_SIMILAR_GRAPHS, PATH_TO_STATISTICS

from eval import compute_random_baseline, mean_average_precision
warnings.filterwarnings("ignore")


def get_amr_sentences(lines: List[str]):
    results = []
    i = 0
    while i != len(lines):
        if lines[i].startswith("# ::snt"):
            start = i
            i += 1
            while lines[i] != '\n':
                i += 1
            results.append([start, i])
        else:
            i += 1
    amr_sentences = [
        ''.join(lines[start:end])
        for start, end
        in results
    ]
    return amr_sentences


def get_raw_sentences(lines: List[str]):
    results = []
    i = 0
    while i != len(lines):
        if lines[i].startswith("# ::snt"):
            start = i
            i += 1
            while not re.match(r'\([\S]+ \/ [\S]+\n', lines[i]):
                i += 1
            results.append([start, i])
        else:
            i += 1
    raw_sentences = [
        ''.join(lines[start:end])
        for start, end
        in results
    ]
    return raw_sentences


def read_amr_graph(path: Path) -> List[AMR_Container]:
    with open(path, 'r') as f:
        lines = f.readlines()

    amr_sentences = get_amr_sentences(
        lines=lines
    )
    raw_sentences = get_raw_sentences(
        lines=lines
    )

    graphs = []
    n = 0
    for raw_sentence, amr_sentence in tqdm(zip(raw_sentences, amr_sentences), leave=False):
        try:
            # TODO: check why some cases fail to generate the graphviz instance
            # FIXME: the sentences which contain \n (related to the part that we have multiple sentences in one)
            graphs.append(
                AMR_Container(
                    sentence=raw_sentence,
                    graph_str=amr_sentence
                )
            )
        except Exception as e:
            n += 1
            continue
    print(n, len(raw_sentences), len(amr_sentences))
    return graphs


def get_amr_labels_from_csv_file(csv_path: Path or str) -> None:
    if os.path.exists(PATH_TO_MASKED_SENTENCES_AMRS):
        results = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS)
        return results

    df = pd.read_csv(csv_path)

    results = []

    for _, (_, data) in tqdm(enumerate(df.iterrows()), leave=False):

        label = data["updated_label"]
        masked_article = data["masked_articles"]
        original_article = data["source_article"]

        updated_masked_article = re.sub(
            r"MSK<(\d+)>", r"MSK\1", masked_article
        )

        updated_masked_article = re.sub(r"\n", ". ", updated_masked_article)

        amr_container = AMR_Container(
            sentence=updated_masked_article
        )
        results.append([
            original_article,
            amr_container,
            label
        ])
    joblib.dump(
        results,
        PATH_TO_MASKED_SENTENCES_AMRS
    )


if __name__ == "__main__":

    # get_similar_graphs(
    #     index = 10,
    #     graph_embeddings=pd.read_csv(PATH_TO_GRAPH_EMBEDDINGS),
    #     sentences_with_amr_container=joblib.load(PATH_TO_MASKED_SENTENCES_AMRS),
    # )

    # top_ns = [5, 10, 20]

    results = pd.read_csv(PATH_TO_MOST_SIMILAR_GRAPHS)
    fallacy_types_counts = results[['sent_a', 'type_a']].drop_duplicates(
    ).groupby("type_a").apply(lambda x: len(x))
    statistics = pd.DataFrame()
    top_n = 10

    all_types = results['type_a'].unique().tolist()
    for type in all_types:
        type_records = results[results['type_a'] == type]
        type_records = type_records.sort_values(by=['sent_a', 'similarity'])

        sub_type_records = type_records.groupby('sent_a').apply(
            lambda x: x[:top_n]).reset_index(drop=True)

        num_sentences = sub_type_records['sent_a'].nunique()
        match_for_each_sentence_vec = sub_type_records.groupby('sent_a').apply(
            lambda x: (np.array(x['type_b'].tolist()) == np.array([type])).astype(int)).values

        statistics = statistics.append({
            'type': type,
            'num_records': num_sentences,
            'top_n': top_n,
            'ratio/all_classes': num_sentences / np.sum(fallacy_types_counts),
            'MAP': mean_average_precision(match_for_each_sentence_vec),
            'random_baseline_MAP': compute_random_baseline(fallacy_types_counts, type, top_n)
        }, ignore_index=True)

    statistics.to_csv(PATH_TO_STATISTICS, index=False)
