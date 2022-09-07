from transformers import BertModel, BertTokenizer
import torch as th
from IPython import embed
import joblib
import re

from consts import PATH_TO_MASKED_SENTENCES_AMRS

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
# load
model = BertModel.from_pretrained(model_name)


def get_bert_embeddings(word: str) -> th.Tensor:
    input_text = word
    input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors = 'pt')

    outputs = model(input_ids)
    return outputs['last_hidden_state'].squeeze()
    


if __name__ == "__main__":
    amr_graphs_with_sentences = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS)
    sample_graph = amr_graphs_with_sentences[0][1].graph_nx
    
    

    
        



        

