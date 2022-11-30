import argparse
import os
import sys
from sentence_transformers import SentenceTransformer
import joblib
from IPython import embed
import pandas as pd
import torch

from sklearn.metrics.pairwise import cosine_similarity


def generate_the_similarities(source_file: str, target_file: str, output_file: str, checkpoint: str):
    if os.path.exists(output_file):
        print(f"Output file already exists for {target_file}. Skipping...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer(checkpoint)

    train_sentences = pd.read_csv(source_file)["text"].tolist()
    train_sentences = [x.strip() for x in train_sentences]

    all_sentences = pd.read_csv(target_file)["text"].tolist()
    all_sentences = [x.strip() for x in all_sentences]

    print(f"Calculating similarities for {target_file}...")

    train_embeddings = model.encode(
        train_sentences, show_progress_bar=True, device=device)
    all_sentences_embeddings = model.encode(
        all_sentences, show_progress_bar=True, device=device)

    all_to_train_similarities = cosine_similarity(
        all_sentences_embeddings, train_embeddings)

    similarities_dict = dict()
    for sentence, row in zip(all_sentences, all_to_train_similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    generate_the_similarities(
        args.source_file, args.target_file, args.output_file, args.checkpoint)
