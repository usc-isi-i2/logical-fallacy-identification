import argparse
import os
import sys

import joblib
from simcse import SimCSE

this_dir = os.path.dirname(__file__)  # Path to loader.py
sys.path.append(os.path.join(this_dir, "../../amr/"))


def get_source_feature_from_amr_objects(sentences_with_amr_objects, source_feature):
    if source_feature == "source_article":
        return [obj[0].strip()
                for obj in sentences_with_amr_objects]
    elif source_feature == "masked_articles":
        return [obj[1].sentence.strip()
                for obj in sentences_with_amr_objects]


def generate_the_simcse_similarities(source_file: str, source_feature: str, target_file: str, output_file: str):
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    sentences_with_amr_objects = joblib.load(source_file)

    train_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, source_feature)

    sentences_with_amr_objects = joblib.load(target_file)
    all_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, source_feature)
    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_feature',
                        choices=['masked_articles', 'source_article'], type=str)
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
