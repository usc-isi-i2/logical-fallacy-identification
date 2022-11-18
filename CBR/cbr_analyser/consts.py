

from pathlib import Path


PATH_TO_GRAPH_EMBEDDINGS = (
    (Path(__file__).parent) / "../cache/graph_2_vec_results.csv").absolute()

PATH_TO_MOST_SIMILAR_GRAPHS = (
    (Path(__file__).parent) / "../cache/most_similar_graphs_for_each.").absolute()

PATH_TO_STATISTICS = ((Path(__file__).parent) /
                      "../cache/statistics.csv").absolute()


good_relations = ['/r/Causes', '/r/UsedFor', '/r/CapableOf', '/r/CausesDesire', '/r/IsA', '/r/SymbolOf', '/r/MadeOf',
                  '/r/LocatedNear', '/r/Desires', '/r/AtLocation', '/r/HasProperty', '/r/PartOf', '/r/HasFirstSubevent', '/r/HasLastSubevent']

good_relations_labels = [
    'at location',
    'capable of',
    'causes',
    'causes desire',
    'desires',
    'has first subevent',
    'has last subevent',
    'has property',
    'is a',
    'located near',
    'made of',
    'part of',
    'symbol of',
    'used for'
]


num_edge_types = 111


datasets_config = {
    "data/logical_fallacy": {
        'features': {
            'text': 'source_article',
            'label': 'updated_label'
        },
        'classes': {
            'ad hominem': 0,
            'ad populum': 1,
            'appeal to emotion': 2,
            'circular reasoning': 3,
            'fallacy of credibility': 4,
            'fallacy of extension': 5,
            'fallacy of logic': 6,
            'fallacy of relevance': 7,
            'false causality': 8,
            'false dilemma': 9,
            'faulty generalization': 10,
            'intentional': 11,
            'equivocation': 12
        }
    },
    "data/finegrained": {
        'features': {
            'text': 'text',
            'label': 'label'
        },
        'classes': {
            "fallacy of relevance": 0,
            "faulty generalization": 1,
            "ad hominem": 2,
            "false causality": 3,
            "circular reasoning": 4,
            "ad populum": 5,
            "fallacy of credibility": 6,
            "appeal to emotion": 7,
            "fallacy of logic": 8,
            "intentional": 9,
            "false dilemma": 10,
            "fallacy of extension": 11,
            "equivocation": 12,
        }
    },
    "data/coarsegrained": {
        'features': {
            'text': 'text',
            'label': 'label'
        },
        'classes': {
            "fallacy of relevance": 0,
            "fallacies of defective induction": 1,
            "fallacies of presumption": 2,
            "fallacy of ambiguity": 3,
        }
    },
    "data/bigbench": {
        'features': {
            'text': 'text',
            'label': 'label'
        },
        'classes': {
            'negative': 0,
            'positive': 1
        }
    }
}

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]
