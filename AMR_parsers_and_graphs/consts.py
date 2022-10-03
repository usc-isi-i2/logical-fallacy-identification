from pathlib import Path


PATH_TO_ORIGINAL_TRAIN_DATA = (
    (Path(__file__).parent) / "data/edu_train.csv").absolute()
PATH_TO_ORIGINAL_DEV_DATA = (
    (Path(__file__).parent) / "data/edu_dev.csv").absolute()
PATH_TO_ORIGINAL_TEST_DATA = (
    (Path(__file__).parent) / "data/edu_test.csv").absolute()


PATH_TO_EXPLAGRAPH_DEV_FILE = ((Path(__file__).parent) /
                               "tmp/explagraph/dev.tsv").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TRAIN = ((Path(__file__).parent) /
                                       "tmp/masked_sentences_with_AMR_container_objects_train.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TRAIN_WITH_BELIEF_ARGUMENT = ((Path(__file__).parent) /
                                                            "tmp/masked_sentences_with_AMR_container_objects_train_with_belief_argument.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_DEV_WITH_BELIEF_ARGUMENT = ((Path(__file__).parent) /
                                                          "tmp/masked_sentences_with_AMR_container_objects_dev_with_belief_argument.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TEST_WITH_BELIEF_ARGUMENT = ((Path(__file__).parent) /
                                                           "tmp/masked_sentences_with_AMR_container_objects_test_with_belief_argument.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_DEV = ((Path(__file__).parent) /
                                     "tmp/masked_sentences_with_AMR_container_objects_dev.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS_DEV_WORDNET = ((Path(__file__).parent) /
                                             "tmp/masked_sentences_with_AMR_container_objects_dev_wordnet.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TEST_WORDNET = ((Path(__file__).parent) /
                                              "tmp/masked_sentences_with_AMR_container_objects_test_wordnet.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_DEV_CONCEPTNET = ((Path(__file__).parent) /
                                                "tmp/masked_sentences_with_AMR_container_objects_dev_conceptnet.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TEST_CONCEPTNET = ((Path(__file__).parent) /
                                                 "tmp/masked_sentences_with_AMR_container_objects_test_conceptnet.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TEST = ((Path(__file__).parent) /
                                      "tmp/masked_sentences_with_AMR_container_objects_test.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS_TRAIN_WORDNET = ((Path(__file__).parent) /
                                               "tmp/masked_sentences_with_AMR_container_objects_train_wordnet.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS_TRAIN_CONCEPTNET = ((Path(__file__).parent) /
                                                  "tmp/masked_sentences_with_AMR_container_objects_train_conceptnet.joblib").absolute()


PATH_TO_GRAPH_EMBEDDINGS = (
    (Path(__file__).parent) / "tmp/graph_2_vec_results.csv").absolute()

PATH_TO_MOST_SIMILAR_GRAPHS = (
    (Path(__file__).parent) / "tmp/most_similar_graphs_for_each.").absolute()

PATH_TO_STATISTICS = ((Path(__file__).parent) /
                      "tmp/statistics.csv").absolute()

PATH_TO_SIMCSE_SIMILARITIES_TRAIN = ((Path(__file__).parent) /
                                     "tmp/simcse_similarities_train.joblib").absolute()

PATH_TO_SIMCSE_SIMILARITIES_DEV = ((Path(__file__).parent) /
                                   "tmp/simcse_similarities_dev.joblib").absolute()

PATH_TO_SIMCSE_SIMILARITIES_TEST = ((Path(__file__).parent) /
                                    "tmp/simcse_similarities_test.joblib").absolute()

PATH_TO_SENTENCE_TRANSFORMERS_SIMILARITIES_TRAIN = ((Path(__file__).parent) /
                                                    "tmp/sentence_transformers_similarities_train.joblib").absolute()

PATH_TO_SENTENCE_TRANSFORMERS_SIMILARITIES_DEV = ((Path(__file__).parent) /
                                                  "tmp/sentence_transformers_similarities_dev.joblib").absolute()

PATH_TO_SENTENCE_TRANSFORMERS_SIMILARITIES_TEST = ((Path(__file__).parent) /
                                                   "tmp/sentence_transformers_similarities_test.joblib").absolute()

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


label2index = {
    'ad hominem': 0,
    'ad populum': 1,
    'appeal to emotion': 2,
    'circular reasoning': 3,
    'equivocation': 4,
    'fallacy of credibility': 5,
    'fallacy of extension': 6,
    'fallacy of logic': 7,
    'fallacy of relevance': 8,
    'false causality': 9,
    'false dilemma': 10,
    'faulty generalization': 11,
    'intentional': 12
}


index2label = {v: k for k, v in label2index.items()}
