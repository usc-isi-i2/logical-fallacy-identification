from pathlib import Path


PATH_TO_ORIGINAL_TRAIN_DATA = ((Path(__file__).parent) / "data/edu_train.csv").absolute()
PATH_TO_ORIGINAL_DEV_DATA = ((Path(__file__).parent) / "data/edu_dev.csv").absolute()
PATH_TO_ORIGINAL_TEST_DATA = ((Path(__file__).parent) / "data/edu_test.csv").absolute()


PATH_TO_SENTENCES_AMR_OBJECTS = ((Path(__file__).parent) /
                                 "tmp/sentences_with_AMR_container_objects.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_DEV = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_dev.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_DEV_WITH_LABEL2WORD = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_dev_with_label2words.joblib").absolute()                                                                 

PATH_TO_MASKED_SENTENCES_AMRS_TEST = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_test.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TEST_WITH_LABEL2WORD = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_test_with_label2words.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_with_label2words.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_with_label2words_wordnet.joblib").absolute()


PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET_CONCEPTNET = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects_with_label2words_wordnet_conceptnet.joblib").absolute()

PATH_TO_MASKED_SENTENCES_AMRS_TMP = ((Path(__file__).parent) /
                                 "tmp/masked_sentences_with_AMR_container_objects.tmp.joblib").absolute()


PATH_TO_GRAPH_EMBEDDINGS = (
    (Path(__file__).parent) / "tmp/graph_2_vec_results.csv").absolute()

PATH_TO_MOST_SIMILAR_GRAPHS = (
    (Path(__file__).parent) / "tmp/most_similar_graphs_for_each.").absolute()

PATH_TO_STATISTICS = ((Path(__file__).parent) /
                      "tmp/statistics.csv").absolute()
