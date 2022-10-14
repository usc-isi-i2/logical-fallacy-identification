import re

import joblib
import torch as th
from IPython import embed
from transformers import BertModel, BertTokenizer

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
# load
model = BertModel.from_pretrained(model_name)


def get_bert_embeddings(word: str) -> th.Tensor:
    try:
        input_text = word
        input_ids = tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors='pt')

        outputs = model(input_ids)
        outputs = outputs['last_hidden_state']
    except Exception as e:
        embed()
    return th.mean(outputs, dim=(0, 1))


if __name__ == "__main__":
    word = "degree"
    embedding = get_bert_embeddings(word)
    print(embedding.shape)
    embed()
    # amr_graphs_with_sentences = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS)
    # sample_graph = amr_graphs_with_sentences[0][1].graph_nx
