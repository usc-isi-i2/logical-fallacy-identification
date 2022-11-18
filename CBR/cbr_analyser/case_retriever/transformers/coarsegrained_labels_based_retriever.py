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

from typing import Any, Dict
import wandb

bad_classes = [
    'prejudicial language',
    'fallacy of slippery slope',
    'slothful induction'
]

def do_train_process(config = None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
        
        train_df = pd.read_csv(os.path.join(config.data_dir, 'train.csv'))
        dev_df = pd.read_csv(os.path.join(config.data_dir, 'dev.csv'))
        test_df = pd.read_csv(os.path.join(config.data_dir, 'test.csv'))
        
        train_df = train_df[~train_df['label'].isin(bad_classes)]
        dev_df = dev_df[~dev_df['label'].isin(bad_classes)]
        test_df = test_df[~test_df['label'].isin(bad_classes)]
 

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['coarse_label'])

        train_df['coarse_label'] = label_encoder.transform(
            train_df['coarse_label'])
        dev_df['coarse_label'] = label_encoder.transform(dev_df['coarse_label'])
        test_df['coarse_label'] = label_encoder.transform(
            test_df['coarse_label'])

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(dev_df),
            'test': Dataset.from_pandas(test_df)
        })

        def process(batch):
            texts = batch['text']
            inputs = tokenizer(texts, truncation=True)
            return {
                **inputs,
                'labels': batch['coarse_label']
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


        model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/nli-roberta-base', num_labels=len(list(label_encoder.classes_)), classifier_dropout = config.classifier_dropout, ignore_mismatched_sizes=True)

        print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir='./finegrained_labels_based_retriever',
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy='steps',
            report_to='wandb'
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

if __name__ == '__main__':

    sweep_config = {
        'method': 'random',
    }

    metric = {
        'name': 'eval/f1',
        'goal': 'maximize'   
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'data_dir': {
            'values': ['data/finegrained']
        },
        'batch_size': {
            'values': [8]
        },
        'learning_rate': {
            # 'distribution': 'uniform',
            # 'min': 5e-6 if args.data_dir == 'data/finegrained' else 1e-6,
            # 'max': 5e-5 if args.data_dir == 'data/finegrained' else 1e-5,
            'values': [5e-5]
        },
        'num_epochs': {
            'values':[15]
        },
        'classifier_dropout': {
            # 'values': [0.1, 0.3]
            'values': [0.1]
        },
        'weight_decay': {
            # 'distribution': 'uniform',
            # 'min': 1e-4,
            # 'max': 1e-1
            'values': [0.01]
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='Finegrained dataset on coarsegrained labels')
    wandb.agent(sweep_id, do_train_process, count=1)


    