import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import joblib
import pandas as pd
import wandb
from typing import List, Dict
from pathlib import Path
from transformers import TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from IPython import embed
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import argparse
import networkx as nx


NUM_LABELS = 13
BATCH_SIZE = 16
NUM_TRAINING_EPOCHS = 7
DROPOUT_PROB = 0.1


parser = argparse.ArgumentParser(
    description='Train a Classification Model for Logical Fallacy Detection')
parser.add_argument(
    '--experiment', help='The experiment we want to do by doing the training', type=str, default="train")
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
    '--input_type', help="The type of the input getting fed to the model (csv or amr)"
)

parser.add_argument(
    '--augments', help="What part of the augmentation to be included in the sentences", type=str
)

args = parser.parse_args()

# TODO: Check the comparison between the original and masked articles
# TODO: Add similar sentences when training and see what happens


def get_wordnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
    template_dict = {
        "syn": " is synonym of ",
        "ant": " is antonym of ",
        "entails": " entails ",
        "part_of": " is part of "
    }
    results = []
    for edge in graph.edges(data=True):
        if edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
            results.append(
                f"{node2label[edge[0]]}{template_dict[edge[2]['label']]}{node2label[edge[1]]}"
            )

    return results


def get_conceptnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
    # TODO: implemented this
    raise NotImplementedError()


def read_csv_from_amr(input_file: str, augments=[]) -> pd.DataFrame:
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
        masked_sentence = obj[1].sentence
        graph = obj[1].graph_nx
        augmented_sentences = []
        if "wordnet" in augments:
            augmented_sentences.extend(
                get_wordnet_edges_in_sentences(
                    graph,
                    obj[1].label2word
                )
            )
        if "conceptnet" in augments:
            augmented_sentences.extend(
                get_conceptnet_edges_in_sentences(
                    graph,
                    obj[1].label2word
                )
            )
        sentences.append(
            "; ".join([masked_sentence, *augmented_sentences])
        )
        labels.append(obj[2])

    return pd.DataFrame({
        "masked_articles": sentences,
        "updated_label": labels
    })


def augment_records_with_similar_records(data_df: pd.DataFrame, num_cases: int = 3) -> pd.DataFrame:
    """Add similar sentences to each record

    Args:
        data_df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: output DataFrame augmented with similar entries
    """
    new_sentences = []
    new_labels = []
    for _, info in data_df.iterrows():
        base_sentence = info['masked_articles']
        similar_cases = data_df[data_df['updated_label'] == info['updated_label']].sample(num_cases)[
            'masked_articles'].tolist()
        new_sentence = ";".join([base_sentence, *similar_cases])
        new_sentences.append(new_sentence)
        new_labels.append(info['updated_label'])
    return pd.DataFrame({
        'masked_articles': new_sentences,
        'updated_label': new_labels
    })


# READING THE DATA

if args.input_type == "csv":
    train_df = pd.read_csv(args.train_input_file)[
        ['masked_articles', 'updated_label']]
    dev_df = pd.read_csv(args.dev_input_file)[
        ['masked_articles', 'updated_label']]
    test_df = pd.read_csv(args.test_input_file)[
        ['masked_articles', 'updated_label']]

elif args.input_type == "amr":
    train_df = read_csv_from_amr(
        input_file=args.train_input_file, augments=args.augments.split('&'))
    dev_df = read_csv_from_amr(
        input_file=args.dev_input_file, augments=args.augments.split('&'))
    test_df = read_csv_from_amr(
        input_file=args.test_input_file, augments=args.augments.split('&'))

# WHETHER TO AUGMENT THE DATA OR NOT

if args.experiment == "case_augmented_training":
    train_df = augment_records_with_similar_records(train_df)
    dev_df = augment_records_with_similar_records(dev_df)
    test_df = augment_records_with_similar_records(test_df)


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
    "roberta-large", num_labels=NUM_LABELS, hidden_dropout_prob=DROPOUT_PROB, attention_probs_dropout_prob=DROPOUT_PROB)

print('Model loaded!')

wandb.init(project="logical_fallacy_classification", entity="zhpinkman", config={
    "batch_size": BATCH_SIZE,
    "num_training_epochs": NUM_TRAINING_EPOCHS,
    "experiment": args.experiment,
    "augments": args.augments,
    'input_type': args.input_type,
    "dropout_prob": DROPOUT_PROB
})

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
