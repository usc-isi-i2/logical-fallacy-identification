import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import joblib
import pandas as pd
import wandb
import gcn
from typing import List, Dict
from pathlib import Path
from transformers import TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from IPython import embed
import re
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import argparse
import networkx as nx


NUM_LABELS = 13
BATCH_SIZE = 16
NUM_TRAINING_EPOCHS = 8
DROPOUT_PROB = 0.1
LEARNING_RATE = 1e-5

# TODO: Check the comparison between the original and masked articles
# TODO: Add similar sentences when training and see what happens


def get_wordnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
    template_dict = {
        "syn": " is synonym of ",
        "ant": " is antonym of ",
        "entails": " entails ",
        "part_of": " is part of "
    }
    results = set()
    for edge in graph.edges(data=True):
        if edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
            results.add(
                f"{node2label[edge[0]]}{template_dict[edge[2]['label']]}{node2label[edge[1]]}"
            )

    return results


def get_conceptnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
    def convert_camelCase_to_space(label):
        label = re.sub(
            r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', label)
        return label
    results = set()
    for edge in graph.edges(data=True):
        if not edge[2]['label'].startswith('"') and not edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
            results.add(
                f"{node2label[edge[0]]} {convert_camelCase_to_space(edge[2]['label'])} {node2label[edge[1]]}"
            )
    return results


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
            "; ".join([masked_sentence, *augmented_sentences])
        )
        labels.append(obj[2])

    return pd.DataFrame({
        "masked_articles": sentences,
        "updated_label": labels
    })


def augment_records_with_similar_records(
    data_df: pd.DataFrame,
    source_df: pd.DataFrame,
    gcn_predictions,
    gcn_all_sentences,
    gcn_all_confidences,
    num_cases: int = 3
) -> pd.DataFrame:
    """Add similar sentences to each record

    Args:
        data_df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: output DataFrame augmented with similar entries
    """

    prediction_table = dict()
    for sentence, prediction, confidence in zip(gcn_all_sentences, gcn_predictions, gcn_all_confidences):
        prediction_table[sentence] = (prediction, confidence)

    all_predictions = []
    all_true_labels = []

    new_sentences = []
    new_labels = []
    num_with_high_confidence = 0

    for _, info in data_df.iterrows():
        try:
            base_sentence = info['masked_articles']

            predicted_label, conf = prediction_table[base_sentence]
            true_label = info['updated_label']
            all_predictions.append(predicted_label)
            all_true_labels.append(true_label)
            if conf > 0.5:
                num_with_high_confidence += 1
                similar_cases = source_df[source_df['updated_label'] == true_label].sample(num_cases)[
                    'masked_articles'].tolist()
            new_sentence = ";".join([base_sentence, *similar_cases])
            new_sentences.append(new_sentence)
            new_labels.append(info['updated_label'])
        except Exception as e:
            print(
                f"Error finding the predicted label for sentence: {base_sentence}")
            pass
    print(classification_report(
        y_pred=all_predictions,
        y_true=all_true_labels
    ))
    print(
        f'Number of data points augmented with high confidence: {num_with_high_confidence / len(data_df)}')
    return pd.DataFrame({
        'masked_articles': new_sentences,
        'updated_label': new_labels
    })


# READING THE DATA

if __name__ == "__main__":
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
        '--augments', help="What part of the augmentation to be included in the sentences", type=str, default=""
    )

    parser.add_argument(
        '--gcn_model_path', help="Path to the pre-trained Graph Convolutional Network used for case retrieval"
    )

    parser.add_argument(
        '--num_cases', help="Number of cases used in Case-based reasoning", type=int, default=0
    )

    args = parser.parse_args()

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

        print('doing the case augmentation ...')
        gcn_model = gcn.CBRetriever(
            num_input_features=gcn.NODE_EMBEDDING_SIZE,
            num_output_features=len(gcn.label2index),
            mid_layer_dropout=gcn.MID_LAYER_DROPOUT,
            mid_layer_embeddings=gcn.LAYERS_EMBEDDINGS,
            heads=4
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gcn_model.load_state_dict(torch.load(args.gcn_model_path))
        gcn_model = gcn_model.to(device)

        train_dataset = gcn.Logical_Fallacy_Dataset(
            path_to_dataset=args.train_input_file,
            fit=True
        )
        dev_dataset = gcn.Logical_Fallacy_Dataset(
            path_to_dataset=args.dev_input_file,
            fit=False,
            all_edge_types=train_dataset.all_edge_types,
            ohe=train_dataset.ohe
        )

        test_dataset = gcn.Logical_Fallacy_Dataset(
            path_to_dataset=args.test_input_file,
            fit=False,
            all_edge_types=train_dataset.all_edge_types,
            ohe=train_dataset.ohe
        )

        train_data_loader = gcn.DataLoader(
            train_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)
        dev_data_loader = gcn.DataLoader(
            dev_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)
        test_data_loader = gcn.DataLoader(
            test_dataset, batch_size=gcn.BATCH_SIZE, shuffle=False)

        train_predictions, train_confidence, all_train_sentences = gcn.do_predict_process(
            model=gcn_model,
            loader=train_data_loader
        )
        dev_predictions, dev_confidence, all_dev_sentences = gcn.do_predict_process(
            model=gcn_model,
            loader=dev_data_loader,
        )
        test_predictions, test_confidence, all_test_sentences = gcn.do_predict_process(
            model=gcn_model,
            loader=test_data_loader
        )

        del gcn_model

        train_df = augment_records_with_similar_records(
            data_df=train_df,
            source_df=train_df,
            num_cases=args.num_cases,
            gcn_predictions=train_predictions,
            gcn_all_confidences=train_confidence,
            gcn_all_sentences=all_train_sentences
        )
        dev_df = augment_records_with_similar_records(
            data_df=dev_df,
            source_df=train_df,
            num_cases=args.num_cases,
            gcn_predictions=dev_predictions,
            gcn_all_confidences=dev_confidence,
            gcn_all_sentences=all_dev_sentences
        )
        test_df = augment_records_with_similar_records(
            data_df=test_df,
            source_df=train_df,
            num_cases=args.num_cases,
            gcn_predictions=test_predictions,
            gcn_all_confidences=test_confidence,
            gcn_all_sentences=all_test_sentences
        )

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
        "roberta-large", num_labels=NUM_LABELS)

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
        learning_rate=LEARNING_RATE,
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
