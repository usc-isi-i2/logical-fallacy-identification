import numpy as np
import pandas as pd


def mean_reciprocal_rank(rs) -> float:
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def compute_random_baseline(fallacy_types_numbers: pd.Series, fallacy_type: str, top_n: int = 10):
    type_prob_dict = dict(zip(
        fallacy_types_numbers.index,
        fallacy_types_numbers.values / np.sum(fallacy_types_numbers)
    ))

    results = []
    for _ in range(fallacy_types_numbers[fallacy_type]):
        predictions = np.random.choice(
            a=list(type_prob_dict.keys()),
            size=top_n,
            p=list(type_prob_dict.values())
        )
        predictions = (np.array([type]) == np.array(predictions)).astype(int)
        results.append(predictions)
    return mean_average_precision(np.array(results))
