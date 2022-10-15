import argparse
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from IPython import embed
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import cbr_analyser.case_retriever.gcn as gcn
import cbr_analyser.consts as consts
import wandb

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))

torch.manual_seed(77)
random.seed(77)
np.random.seed(77)

NUM_LABELS = 13




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

def augment_with_cases_similarity_matrices(train_df, dev_df, test_df, args, sep_token):
    if args["all_bad_cases"]:
        all_bad_cases = joblib.load(args["all_bad_cases"])
    if args["all_good_cases"]:
        all_good_cases = joblib.load(args["all_good_cases"])
    simcse_train_similarities = joblib.load(args["similarity_matrices_path_train"])
    simcse_dev_similarities = joblib.load(args["similarity_matrices_path_dev"])
    simcse_test_similarities = joblib.load(args["similarity_matrices_path_test"])
    
    
    def augment_dataframe(df, simcse_similarities):
        augmented_sentences = []
        external_sentences = []
        for sentence in df[args["source_feature"]]:
            try:
                sentences_and_similarities = simcse_similarities[sentence.strip()].items()
                sentences_and_similarities_sorted = sorted(sentences_and_similarities, key = lambda x: x[1], reverse = True)
                augs = [x[0] for x in sentences_and_similarities_sorted[1:args["num_cases"] + 1] if x[1] > args["cbr_threshold"]]
                if args["all_bad_cases"]:
                    augs = [sent for sent in augs if sent not in all_bad_cases]
                if args["all_good_cases"]:
                    augs = [sent for sent in augs if sent in all_good_cases]
                result_sentence = f"{sentence} {sep_token}{sep_token} {' '.join(augs)}"
                external_sentences.append('</sep>'.join(augs))
                augmented_sentences.append(result_sentence)
            except Exception as e:
                embed()
                
        df[args["source_feature"]] = augmented_sentences
        df['cbr'] = external_sentences
        return df

    train_df = augment_dataframe(train_df, simcse_train_similarities)
    dev_df = augment_dataframe(dev_df, simcse_dev_similarities)
    test_df = augment_dataframe(test_df, simcse_test_similarities)
    
    return train_df, dev_df, test_df

    

# def augment_with_cases_gcn(train_df, dev_df, test_df, args):
#     print('doing the case augmentation using GCN ...')
#     gcn_model = gcn.CBRetriever(
#         num_input_features=gcn.NODE_EMBEDDING_SIZE,
#         num_output_features=len(consts.label2index),
#         mid_layer_dropout=gcn.MID_LAYERS_DROPOUT,
#         mid_layer_embeddings=[int(x)
#                               for x in gcn.MID_LAYERS_EMBEDDINGS.split('&')],
#         heads=4
#     )

#     device_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if device_cuda else "cpu")
    
#     if not device_cuda:
#         gcn_model.load_state_dict(torch.load(
#             args["gcn_model_path"], map_location=torch.device('cpu')))
#     else:
#         gcn_model.load_state_dict(torch.load(args["gcn_model_path"]))
#     gcn_model = gcn_model.to(device)

#     train_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args["train_input_file"],
#         fit=True
#     )
#     dev_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args["dev_input_file"],
#         fit=False,
#         all_edge_types=train_dataset.all_edge_types,
#         ohe=train_dataset.ohe
#     )

#     test_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args["test_input_file"],
#         fit=False,
#         all_edge_types=train_dataset.all_edge_types,
#         ohe=train_dataset.ohe
#     )

#     train_data_loader = gcn.DataLoader(
#         train_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)
#     dev_data_loader = gcn.DataLoader(
#         dev_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)
#     test_data_loader = gcn.DataLoader(
#         test_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)

#     train_results = gcn.test_on_loader(
#         gcn_model, train_data_loader)
#     dev_results = gcn.test_on_loader(
#         gcn_model, dev_data_loader)
#     test_results = gcn.test_on_loader(
#         gcn_model, test_data_loader)

#     all_train_predictions = [consts.index2label[pred]
#                              for pred in train_results['predictions']]
#     all_dev_predictions = [consts.index2label[pred]
#                            for pred in dev_results['predictions']]
#     all_test_predictions = [consts.index2label[pred]
#                             for pred in test_results['predictions']]
#     all_train_true_labels = [consts.index2label[true_label]
#                              for true_label in train_results['true_labels']]
#     all_dev_true_labels = [consts.index2label[true_label]
#                            for true_label in dev_results['true_labels']]
#     all_test_true_labels = [consts.index2label[true_label]
#                             for true_label in test_results['true_labels']]

#     print('Train split results')
#     print(classification_report(
#         y_pred=all_train_predictions,
#         y_true=all_train_true_labels
#     ))

#     print('Dev split results')
#     print(classification_report(
#         y_pred=all_dev_predictions,
#         y_true=all_dev_true_labels
#     ))

#     print('Test split results')
#     print(classification_report(
#         y_pred=all_test_predictions,
#         y_true=all_test_true_labels
#     ))


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

        
        

    

def do_train_process(args):
    tokenizer = AutoTokenizer.from_pretrained(args["checkpoint"])
    if args["checkpoint"] != "bigscience/bloom-560m":
        model = AutoModelForSequenceClassification.from_pretrained(
            args["checkpoint"], num_labels=NUM_LABELS, classifier_dropout = args["classifier_dropout"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args["checkpoint"], num_labels=NUM_LABELS)

    train_df = read_csv_from_amr(input_file=args["train_input_file"], augments=args["augments"].split('&'), source_feature=args["source_feature"])
    dev_df = read_csv_from_amr(input_file=args["dev_input_file"], augments=args["augments"].split('&'), source_feature=args["source_feature"])
    test_df = read_csv_from_amr(input_file=args["test_input_file"], augments=args["augments"].split('&'), source_feature=args["source_feature"])

    if args["cbr"]:
        train_df, dev_df, test_df = augment_with_cases_similarity_matrices(train_df, dev_df, test_df, args, tokenizer.sep_token)


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
        '--augments', type = str, default = ""
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


    