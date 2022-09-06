from pathlib import Path


PATH_TO_TRAIN_DATA = ((Path(__file__).parent) /
                      "data/edu_train_amr_parse_tree.txt").absolute()


PATH_TO_SENTENCES_AMR_OBJECTS = ((Path(__file__).parent) /
                                 "tmp/sentences_with_AMR_container_objects.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS = ((Path(__file__).parent) /
                                     "tmp/masked_sentences_with_AMR_container_objects.joblib").absolute()


PATH_TO_GRAPH_EMBEDDINGS = ((Path(__file__).parent) / "tmp/graph_2_vec_results.csv").absolute()

PATH_TO_MOST_SIMILAR_GRAPHS = ((Path(__file__).parent) / "tmp/most_similar_graphs_for_each.csv").absolute()

PATH_TO_STATISTICS = ((Path(__file__).parent) / "tmp/statistics.csv").absolute()
