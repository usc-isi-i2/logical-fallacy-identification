import argparse
import os
import sys

import joblib
import numpy as np
import torch
from IPython import embed
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../../amr/"))


def get_embeddings_empathy(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt",
                       truncation=True, padding=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    outputs = model(**inputs)

    return outputs.pooler_output.detach().cpu()


def get_source_feature_from_amr_objects(sentences_with_amr_objects, source_feature):
    if source_feature == "source_article":
        return [obj[0].strip()
                for obj in sentences_with_amr_objects]
    elif source_feature == "masked_articles":
        return [obj[1].sentence.strip()
                for obj in sentences_with_amr_objects]


def generate_the_empathy_similarities(source_file: str, source_feature: str, target_file: str, output_file: str):

    checkpoint = "bdotloh/roberta-base-empathy"

    tokenizer = RobertaTokenizer.from_pretrained(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaModel.from_pretrained(checkpoint)
    model = model.to(device)

    sentences_with_amr_objects = joblib.load(source_file)

    train_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, source_feature)

    sentences_with_amr_objects = joblib.load(target_file)
    all_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, source_feature)

    similarities = np.zeros([len(all_sentences), len(train_sentences)])

    for i in tqdm(range(len(all_sentences)), leave=False):
        for j in range(0, len(train_sentences), 100):
            embeddings = get_embeddings_empathy(
                tokenizer, model, all_sentences[i:i+1] + train_sentences[j:min(j+100, len(train_sentences))])
            similarities[i, j:min(j+100, len(train_sentences))] = pairwise_cosine_similarity(
                embeddings[0].reshape(1, -1), embeddings[1:]).numpy()

    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":
    pass
