import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed

from sklearn_lvq import GrlvqModel, GmlvqModel
from sklearn_lvq.utils import plot2d
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = RobertaModel.from_pretrained("roberta-base")


def get_embeddings(text: str, tokenizer: RobertaTokenizer, model: RobertaModel) -> np.array:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output.detach().cpu().numpy()
    return pooled_output


def create_dataset(csv_file: str, cap=100):
    df = pd.read_csv(csv_file)
    # df = df[df['updated_label'].isin(['ad populum', 'ad hominem'])]
    texts = df['masked_articles'].tolist()
    labels = df['updated_label'].tolist()

    embeddings = []
    for text in tqdm(texts, leave=False):
        embeddings.append(get_embeddings(text, tokenizer, model))
    return np.array(embeddings), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="edu_train.csv")

    args = parser.parse_args()

    embeddings, labels = create_dataset(args.input_file)
    embeddings = embeddings.squeeze(axis=1)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    labels_encoded = label_encoder.transform(labels).reshape(-1, 1)

    print('GMLVQ:')

    gmlvq = GmlvqModel()
    gmlvq.fit(embeddings, labels)
    plot2d(gmlvq, embeddings, labels, 1, 'gmlvq')

    print('classification accuracy:', gmlvq.score(embeddings, labels))
    plt.show()

    embed()
    exit()
