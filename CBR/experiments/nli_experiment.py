import pandas as pd
import numpy as np
import os
import sys
import joblib
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join('.', "../cbr_analyser/amr/"))

data = joblib.load(
    '../cache/masked_sentences_with_AMR_container_objects_train_with_segments.joblib')

tokenizer = AutoTokenizer.from_pretrained(
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

model = AutoModelForSequenceClassification.from_pretrained(
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = model.to(device)


def extract_relation(chunk1, chunk2):
    max_length = 512

    premise = chunk1
    hypothesis = chunk2

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(
        tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)

    token_type_ids = torch.Tensor(
        tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(
        tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[
        0].tolist()  # batch_size only one

    return {
        "entailment": predicted_probability[0],
        "neutral": predicted_probability[1],
        "contradiction": predicted_probability[2]
    }


sentences_chunks_relations = defaultdict(list)
for obj in tqdm(data, leave=False):
    sentence = obj[0]
    chunks = obj[1].segments
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk1 = chunks[i]
            chunk2 = chunks[j]
            relation = extract_relation(chunk1, chunk2)
            sentences_chunks_relations[sentence].append(
                (chunk1, chunk2, relation))


joblib.dump(sentences_chunks_relations, 'results.joblib')
