import argparse
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from IPython import embed
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from transformers import RobertaTokenizer, RobertaModel
import cbr_analyser.consts as consts
import wandb
from cbr_analyser.case_retriever.retriever import (Empathy_Retriever,
                                                   Retriever, SimCSE_Retriever, GCN_Retriever)
from torch import nn

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))

torch.manual_seed(77)
random.seed(77)
np.random.seed(77)

ROBERTA_HIDDEN_SIZE = 768
NUM_LABELS = 12



def get_wordnet_edges_in_sentences(graph, node2label) -> List[str]:
    template_dict = {
        "syn": "is synonym of",
        "ant": "is antonym of",
        "entails": "entails",
        "part_of": "is part of"
    }
    results = set()
    for edge in graph.edges(data=True):
        if edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
            results.add(
                f"{node2label[edge[0]]} {template_dict[edge[2]['label']]} {node2label[edge[1]]}"
            )

    return results


def get_conceptnet_edges_in_sentences(graph, node2label) -> List[str]:
    results = set()
    for edge in graph.edges(data=True):
        # Edges not from AMR which all of them are in form of "":arg"" and also not from WordNet augmentation
        if not edge[2]['label'].startswith('"') and not edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
            if edge[2]['example'] and type(edge[2]['example']) == str:
                results.add(str(edge[2]['example']).replace('[[', '').replace(']]', ''))
            else:
                results.add(
                    f"{node2label[edge[0]]} {edge[2]['label']} {node2label[edge[1]]}"
                )
    return results


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def read_csv_from_amr(input_file: str, source_feature: str, augments: List[str]=[]) -> pd.DataFrame:
    """Read the sentences alongside their corresponding AMR graphs and output a single DataFrame

    Args:
        input_file (str): input file (.joblib)

    Returns:
        pd.DataFrame: the output
    """
    sentences = []
    labels = []

    data = joblib.load(input_file)
    for obj in data:
        if obj[2] in ['equivocation']:
            continue
        if source_feature == "masked_articles":
            base_sentence = obj[1].sentence
        elif source_feature == "source_article":
            base_sentence = obj[0]
        elif source_feature == "amr_str":
            base_sentence = obj[1].graph_str
        else:
            raise NotImplementedError()
        graph = obj[1].graph_nx
        augmented_sentences = []
        if "wordnet" in augments:
            augmented_sentences.extend(
                list(get_wordnet_edges_in_sentences(
                    graph,
                    obj[1].label2word
                ))
            )
        if "conceptnet" in augments:
            augmented_sentences.extend(
                list(get_conceptnet_edges_in_sentences(
                    graph,
                    obj[1].label2word
                ))
            )
        sentences.append(
            "; ".join([base_sentence, *augmented_sentences])
        )
        labels.append(obj[2])

    return pd.DataFrame({
        source_feature: sentences,
        "updated_label": labels
    })

def augment_with_similar_cases(df: pd.DataFrame, retriever: Retriever, args: Dict[str, Any], sep_token) -> pd.DataFrame:    
    external_sentences = []
    augmented_sentences = []
    count_without_cases = 0
    for sentence in df[args["source_feature"]]:
        try:
            similar_sentences_with_similarities = retriever.retrieve_similar_cases(sentence, args["num_cases"])
            similar_sentences = [s[0] for s in similar_sentences_with_similarities if s[1] > args['cbr_threshold']]
            result_sentence = f"{sentence} {sep_token}{sep_token} {' '.join(similar_sentences)}"
            external_sentences.append('</sep>'.join(similar_sentences))
            augmented_sentences.append(result_sentence)
        except Exception as e:
            print(e)
            count_without_cases += 1
            result_sentence = sentence
            external_sentences.append('')
            augmented_sentences.append(result_sentence)

            
    df[args["source_feature"]] = augmented_sentences
    df['cbr'] = external_sentences
    return df

    

def print_results(label_encoder_inverse, trainer, tokenized_dataset, split: str, args):
    split_predictions = list(map(label_encoder_inverse, np.argmax(trainer.predict(tokenized_dataset[split]).predictions, axis = -1)))
    split_true_labels = list(map(label_encoder_inverse, tokenized_dataset[split]['labels']))

    print(f'performance on {split} data')
    print(classification_report(
        y_pred = split_predictions, 
        y_true = split_true_labels
    ))
    if args["predictions_path"]:
        os.makedirs(args["predictions_path"], exist_ok=True)
        if args["cbr"]:
            results_df = pd.DataFrame({
                'predictions': split_predictions,
                'true_labels': split_true_labels, 
                'cbr': tokenized_dataset[split]['cbr'],
                args["source_feature"]: tokenized_dataset[split][args["source_feature"]]
            })
        else:
            results_df = pd.DataFrame({
                'predictions': split_predictions,
                'true_labels': split_true_labels, 
                args["source_feature"]: tokenized_dataset[split][args["source_feature"]]
            })
        results_df.to_csv(os.path.join(args["predictions_path"], f"{split}.csv"), index = False)

        

def do_train_process(args: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
    if args["checkpoint"] != "bigscience/bloom-560m":
        model = AutoModelForSequenceClassification.from_pretrained(
            args["checkpoint"], num_labels=NUM_LABELS, classifier_dropout = args["classifier_dropout"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args["checkpoint"], num_labels=NUM_LABELS)

    train_df = read_csv_from_amr(input_file=args["train_input_file"], augments=args["augments"], source_feature=args["source_feature"])
    dev_df = read_csv_from_amr(input_file=args["dev_input_file"], augments=args["augments"], source_feature=args["source_feature"])
    test_df = read_csv_from_amr(input_file=args["test_input_file"], augments=args["augments"], source_feature=args["source_feature"])

    if args["cbr"]:
        if args["retriever_type"] == "simcse":
            retriever = SimCSE_Retriever(config = {"source_feature": args["source_feature"]})
        if args['retriever_type'] == "empathy":
            retriever = Empathy_Retriever(config = {"source_feature": args["source_feature"]})
        if args['retriever_type'] == "gcn":
            retriever = GCN_Retriever(config = {"source_feature": args["source_feature"]})
        
        for df in [train_df, dev_df, test_df]:
            df = augment_with_similar_cases(df, retriever, args, tokenizer.sep_token)

    label_encoder = lambda x: consts.label2index[x]
    label_encoder_inverse = lambda x: consts.index2label[x]


    train_df['updated_label'] = list(map(label_encoder, train_df['updated_label']))
    dev_df['updated_label'] = list(map(label_encoder, dev_df['updated_label']))
    test_df['updated_label'] = list(map(label_encoder, test_df['updated_label']))

    

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'eval': Dataset.from_pandas(dev_df),
        'test': Dataset.from_pandas(test_df)
    })

    def process(batch):
        texts = batch[args["source_feature"]]
        inputs = tokenizer(texts, truncation=True)
        return {
            **inputs,
            'labels': batch['updated_label']
        }


    tokenized_dataset = dataset.map(
        process, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    print('Model loaded!')

    training_args = TrainingArguments(
        do_eval=True,
        do_train=True,
        output_dir="./xlm_roberta_logical_fallacy_classification",
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        num_train_epochs=args["num_epochs"],
        overwrite_output_dir = 'True',
        weight_decay=args["weight_decay"],
        logging_steps=50,
        evaluation_strategy='steps',
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print('Start the training ...')
    trainer.train()

    for split in ['train', 'eval', 'test']:
        print_results(
            label_encoder_inverse= label_encoder_inverse, 
            trainer = trainer, 
            tokenized_dataset = tokenized_dataset, 
            split = split,
            args = args
        )
        



# READING THE DATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Classification Model for Logical Fallacy Detection and having a baseline')

    parser.add_argument(
        '--task', type = str, default="train", choices=['train']
    )

    parser.add_argument(
        '--train_input_file', help="Train input file path", type=str
    )
    parser.add_argument(
        '--dev_input_file', help="Dev input file path", type=str
    )
    parser.add_argument(
        '--test_input_file', help="Test input file path", type=str
    )
    parser.add_argument(
        '--source_feature', help="the feature used for training the classification model", type=str
    )
    parser.add_argument(
        '--batch_size', type = int, default = 8
    )

    parser.add_argument(
        '--gcn_model_path', type = str
    )
    parser.add_argument(
        '--model_path'
    )

    parser.add_argument(
        '--learning_rate', type = float, default = 2e-5
    )

    parser.add_argument(
        '--num_epochs', type = int, default = 5
    )

    parser.add_argument(
        '--classifier_dropout', type = float, default = 0.3
    )

    parser.add_argument(
        '--weight_decay', type = float, default = 0.01
    )
    parser.add_argument(
        '--augments', type = lambda x: [] if x is None else x.split('&'), default = [], help = "augments to be used for training the model split by comma"
    )
    parser.add_argument(
        '--cbr', action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--similarity_matrices_path_train', type = str
    )
    
    parser.add_argument(
        '--similarity_matrices_path_dev', type = str
    )
    
    parser.add_argument(
        '--similarity_matrices_path_test', type = str
    )
    
    parser.add_argument(
        '--predictions_path', type = str
    )
    parser.add_argument(
        '--num_cases', type = int, default = 3
    )
    
    parser.add_argument(
        '--all_good_cases', type = str
    )
    parser.add_argument(
        '--all_bad_cases', type = str
    )
    parser.add_argument(
        '--cbr_threshold', type = float, default = -np.inf
    )
    parser.add_argument(
        '--checkpoint', type = str, default = "roberta-base"
    )
    


    args = parser.parse_args()

    wandb.init(
        project="Logical Fallacy Main Classifier",
        entity='zhpinkman',
        config={
            **vars(args)
        }
    )

    if args["task"] == "train":
        do_train_process(args)


    