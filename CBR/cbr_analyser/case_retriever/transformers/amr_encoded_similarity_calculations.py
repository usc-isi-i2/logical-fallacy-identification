import argparse
import os
import sys

import joblib
import numpy as np
import torch
from IPython import embed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import amrlib
from IPython import embed


stog = None


def get_embeddings_amr(texts):
    encodings = stog.encode_sents(texts)

    encodings = torch.cat(encodings, dim=0)

    return encodings.mean(1).squeeze()


def save_all_sentences_embeddings_amr(source_file, output_file):
    print("Loading data")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists")
        return
    global stog
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stog = amrlib.load_stog_model(device=device)

    texts = pd.read_csv(source_file)["text"].tolist()
    texts = [text.strip() for text in texts]

    embeddings = get_embeddings_amr(texts).cpu().numpy()

    output_dict = {
        'sentences': texts,
        'embeddings': embeddings
    }
    print("Saving data")
    joblib.dump(output_dict, output_file)


def generate_the_amr_similarities(source_file: str, target_file: str, output_file: str):

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"File {source_file} does not exist")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"File {target_file} does not exist")

    source_data = joblib.load(source_file)
    target_data = joblib.load(target_file)

    target_sentences = target_data['sentences']
    target_embeddings = target_data['embeddings']

    source_sentences = source_data['sentences']
    source_embeddings = source_data['embeddings']

    similarities = cosine_similarity(source_embeddings, target_embeddings)

    similarities_dict = dict()
    for sentence, similarity in zip(source_sentences, similarities):
        similarities_dict[sentence] = dict(
            zip(target_sentences, similarity.tolist()))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--source_feature", type=str,
                        default="masked_articles")
    parser.add_argument("--target_file", type=str)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    # generate_the_amr_similarities(args.source_file, args.source_feature,
    #                               args.target_file, args.output_file)

    if args.task == "save_embeddings":
        save_all_sentences_embeddings_amr(args.source_file, args.output_file)
    elif args.task == "generate_similarities":
        generate_the_amr_similarities(
            args.source_file, args.target_file, args.output_file)
