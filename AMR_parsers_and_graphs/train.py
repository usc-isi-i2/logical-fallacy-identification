import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from transformers import TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from IPython import embed
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict


TRAIN_DATA_PATH = ((Path(__file__).parent) / "data/edu_train.csv").absolute()
DEV_DATA_PATH = ((Path(__file__).parent) / "data/edu_dev.csv").absolute()
TEST_DATA_PATH = ((Path(__file__).parent) / "data/edu_test.csv").absolute()
NUM_LABELS = 13
BATCH_SIZE = 16
NUM_TRAINING_EPOCHS = 8

train_df = pd.read_csv(TRAIN_DATA_PATH)[['masked_articles', 'updated_label']]
dev_df = pd.read_csv(DEV_DATA_PATH)[['masked_articles', 'updated_label']]
test_df = pd.read_csv(TEST_DATA_PATH)[['masked_articles', 'updated_label']]


label_encoder = LabelEncoder()
label_encoder.fit(train_df['updated_label'])


train_df['updated_label'] = label_encoder.transform(train_df['updated_label'])
dev_df['updated_label'] = label_encoder.transform(dev_df['updated_label'])
test_df['updated_label'] = label_encoder.transform(test_df['updated_label'])


dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'eval': Dataset.from_pandas(dev_df),
    'test': Dataset.from_pandas(test_df)
})


def process(batch):
    texts = batch['masked_articles']
    inputs = tokenizer(texts, truncation=True)
    return {
        **inputs,
        'labels': batch['updated_label']
    }


tokenizer = AutoTokenizer.from_pretrained("roberta-large")
tokenized_dataset = dataset.map(
    process, batched=True, remove_columns=dataset['train'].column_names)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large", num_labels=NUM_LABELS, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)

print('Model loaded!')

wandb.init(project="logical_fallacy_classification", entity="zhpinkman")

training_args = TrainingArguments(
    do_eval=True,
    do_train=True,
    output_dir="./xlm_roberta_logical_fallacy_classification",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_TRAINING_EPOCHS,
    weight_decay=0.01,
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
