import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from cbr_analyser.case_retriever.retriever import (Empathy_Retriever,
                                                   Retriever, SimCSE_Retriever)

from typing import Any, Dict
import wandb

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]

def augment_with_similar_cases(df: pd.DataFrame, retriever: Retriever, config, sep_token, prefix: str, is_train: bool) -> pd.DataFrame:    
    external_sentences = []
    augmented_sentences = []
    count_without_cases = 0
    for sentence in df["text"]:
        try:
            similar_sentences_with_similarities = retriever.retrieve_similar_cases(sentence, config.num_cases)
            similar_sentences = [s[0] for s in similar_sentences_with_similarities if s[1] > config.cbr_threshold]
            result_sentence = f"{sentence} {sep_token}{sep_token} {' '.join(similar_sentences)}"
            external_sentences.append('</sep>'.join(similar_sentences))
            augmented_sentences.append(result_sentence)
        except Exception as e:
            print(e)
            count_without_cases += 1
            result_sentence = sentence
            external_sentences.append('')
            augmented_sentences.append(result_sentence)

            
    df["text"] = augmented_sentences
    df['cbr'] = external_sentences
    return df




def do_train_process(config = None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
        
        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        
        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        
        if config.cbr == True:
            print('using cbr')
            simcse_retriever = SimCSE_Retriever(config = {'data_dir': config.data_dir, 'source_feature': 'masked_articles'})
                        
            for df, is_train in zip([train_df, dev_df, test_df], [True, False, False]):
                df = augment_with_similar_cases(df, simcse_retriever, config, tokenizer.sep_token, "simcse", is_train = is_train)

        
            

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['label'])

        train_df['label'] = label_encoder.transform(
            train_df['label'])
        dev_df['label'] = label_encoder.transform(dev_df['label'])
        test_df['label'] = label_encoder.transform(
            test_df['label'])

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(dev_df),
            'test': Dataset.from_pandas(test_df)
        })

        def process(batch):
            texts = batch["text"]
            inputs = tokenizer(texts, truncation=True)
            return {
                **inputs,
                'labels': batch['label']
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



        model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-roberta-base", num_labels=len(list(label_encoder.classes_)), classifier_dropout = config.classifier_dropout, ignore_mismatched_sizes=True)

        print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir="./xlm_roberta_logical_fallacy_classification",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy='steps',
            report_to="wandb"
        )

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

        print(trainer.predict(tokenized_dataset['test']))


# READING THE DATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Classification Model for Logical Fallacy Detection and having a baseline')

    parser.add_argument(
        '--data_dir', help="Train input file path", type=str
    )

    args = parser.parse_args()

    sweep_config = {
        'method': 'random',
    }

    metric = {
        'name': 'eval/f1',
        'goal': 'maximize'   
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'cbr': {
            "values": [True]
        },
        'num_cases': {
            "values": [1, 2, 3, 4, 5]  
        },
        'cbr_threshold': {
            "values": [-1e7, 0.5, 0.8]
        },
        "data_dir": {
            "values": [args.data_dir]
        },
        "batch_size": {
            "values": [8]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 5e-6 if args.data_dir == "data/finegrained" else 1e-6,
            'max': 5e-5 if args.data_dir == "data/finegrained" else 1e-5,
        },
        "num_epochs": {
            "values":[15]
        },
        "classifier_dropout": {
            "values": [0.1, 0.3]
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 1e-4,
            'max': 1e-1
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="Baseline Finder")
    wandb.agent(sweep_id, do_train_process, count=30)


    