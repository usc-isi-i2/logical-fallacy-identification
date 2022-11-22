import argparse
import os
import sys

import joblib
from IPython import embed
import pandas as pd
from simcse import SimCSE


def get_source_feature_from_amr_objects(sentences_with_amr_objects, source_feature):
    if source_feature == "source_article":
        return [obj[0].strip()
                for obj in sentences_with_amr_objects]
    elif source_feature == "masked_articles":
        return [obj[1].sentence.strip()
                for obj in sentences_with_amr_objects]


def get_embeddings_simcse(model, text: str):
    return model.encode(text)


def generate_the_simcse_similarities(source_file: str, target_file: str, output_file: str):
    if os.path.exists(output_file):
        print(f"Output file already exists for {target_file}. Skipping...")
        return
    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    train_sentences = pd.read_csv(source_file)["text"].tolist()
    train_sentences = [x.strip() for x in train_sentences]

    all_sentences = pd.read_csv(target_file)["text"].tolist()
    all_sentences = [x.strip() for x in all_sentences]

    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    generate_the_simcse_similarities(
        args.source_file, args.target_file, args.output_file)
