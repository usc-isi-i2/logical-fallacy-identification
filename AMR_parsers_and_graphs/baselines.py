from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import wandb
from pathlib import Path
from transformers import TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import argparse


NUM_LABELS = 13


def do_train_process(config = None):
    with wandb.init(config=config):

        config = wandb.config
        
        train_df = pd.read_csv(args.train_input_file)[
            [args.input_feature, 'updated_label']]
        dev_df = pd.read_csv(args.dev_input_file)[
            [args.input_feature, 'updated_label']]
        test_df = pd.read_csv(args.test_input_file)[
            [args.input_feature, 'updated_label']]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['updated_label'])

        train_df['updated_label'] = label_encoder.transform(
            train_df['updated_label'])
        dev_df['updated_label'] = label_encoder.transform(dev_df['updated_label'])
        test_df['updated_label'] = label_encoder.transform(
            test_df['updated_label'])

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(dev_df),
            'test': Dataset.from_pandas(test_df)
        })

        def process(batch):
            texts = batch[args.input_feature]
            inputs = tokenizer(texts, truncation=True)
            return {
                **inputs,
                'labels': batch['updated_label']
            }

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=NUM_LABELS, classifier_dropout = config.classifier_dropout)

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
            logging_steps=50,
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
        '--train_input_file', help="Train input file path", type=str
    )
    parser.add_argument(
        '--dev_input_file', help="Dev input file path", type=str
    )
    parser.add_argument(
        '--test_input_file', help="Test input file path", type=str
    )
    parser.add_argument(
        '--input_feature', help="the feature used for training the classification model", type=str
    )

    args = parser.parse_args()

    sweep_config = {
        'method': 'grid',
    }

    metric = {
        'name': 'eval/f1',
        'goal': 'maximize'   
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        "input_feature": {
            "values": [args.input_feature]
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "learning_rate": {
            "values": [1e-5, 5e-5]
        },
        "num_epochs": {
            "values":[10]
        },
        "classifier_dropout": {
            "values": [0.1, 0.3]
        },
        "weight_decay": {
            "values": [0.01, 0.001]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="Baseline Finder")
    wandb.agent(sweep_id, do_train_process)


    