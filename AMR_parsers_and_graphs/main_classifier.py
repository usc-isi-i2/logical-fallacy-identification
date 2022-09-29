# import torch
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, \
#     classification_report
# import numpy as np
# import joblib
# import pandas as pd
# import wandb
# import gcn
# from typing import List, Dict
# from transformers import TrainingArguments, Trainer, \
#     AutoTokenizer, AutoModelForSequenceClassification
# from transformers import DataCollatorWithPadding
# from IPython import embed
# import re
# from sklearn.preprocessing import LabelEncoder
# from datasets import Dataset, DatasetDict
# import argparse
# import networkx as nx


# def get_wordnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
#     template_dict = {
#         "syn": " is synonym of ",
#         "ant": " is antonym of ",
#         "entails": " entails ",
#         "part_of": " is part of "
#     }
#     results = set()
#     for edge in graph.edges(data=True):
#         if edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
#             results.add(
#                 f"{node2label[edge[0]]}{template_dict[edge[2]['label']]}{node2label[edge[1]]}"
#             )

#     return results


# def get_conceptnet_edges_in_sentences(graph: nx.DiGraph, node2label: Dict[str, str]) -> List[str]:
#     def convert_camelCase_to_space(label):
#         label = re.sub(
#             r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', label)
#         return label
#     results = set()
#     for edge in graph.edges(data=True):
#         if not edge[2]['label'].startswith('"') and not edge[2]['label'] in ["syn", "ant", "entails", 'part_of']:
#             results.add(
#                 f"{node2label[edge[0]]} {convert_camelCase_to_space(edge[2]['label'])} {node2label[edge[1]]}"
#             )
#     return results



# def augment_records_with_similar_records(
#     data_df: pd.DataFrame,
#     source_df: pd.DataFrame,
#     gcn_predictions,
#     gcn_all_sentences,
#     gcn_all_confidences,
#     num_cases: int = 3
# ) -> pd.DataFrame:
#     """Add similar sentences to each record

#     Args:
#         data_df (pd.DataFrame): input DataFrame

#     Returns:
#         pd.DataFrame: output DataFrame augmented with similar entries
#     """

#     prediction_table = dict()
#     for sentence, prediction, confidence in zip(gcn_all_sentences, gcn_predictions, gcn_all_confidences):
#         prediction_table[sentence.strip()] = (prediction, confidence)

#     all_predictions = []
#     all_true_labels = []

#     new_sentences = []
#     new_labels = []
#     num_with_high_confidence = 0

#     for _, info in data_df.iterrows():
#         try:
#             base_sentence = info[args.input_feature].strip()

#             predicted_label, conf = prediction_table[base_sentence]
#             true_label = info['updated_label']
#             all_predictions.append(predicted_label)
#             all_true_labels.append(true_label)
#             if conf > 0.9:
#                 num_with_high_confidence += 1
#                 similar_cases = source_df[source_df['updated_label'] == predicted_label].sample(num_cases)[
#                     args.input_feature].tolist()
#             else:
#                 similar_cases = []
#             new_sentence = ";".join([base_sentence, *similar_cases])
#             new_sentences.append(new_sentence)
#             new_labels.append(info['updated_label'])
#         except Exception as e:
#             embed()
#             print(
#                 f"Error finding the predicted label for sentence: {base_sentence}")
#             pass
#     print(classification_report(
#         y_pred=all_predictions,
#         y_true=all_true_labels
#     ))
#     print(
#         f'Number of data points augmented with high confidence: {num_with_high_confidence / len(data_df)}')
#     return pd.DataFrame({
#         args.input_feature: new_sentences,
#         'updated_label': new_labels
#     })


# if args.experiment == "case_augmented_training":

#     print('doing the case augmentation ...')
#     gcn_model = gcn.CBRetriever(
#         num_input_features=gcn.NODE_EMBEDDING_SIZE,
#         num_output_features=len(gcn.label2index),
#         mid_layer_dropout=gcn.MID_LAYER_DROPOUT,
#         mid_layer_embeddings=gcn.LAYERS_EMBEDDINGS,
#         heads=4
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gcn_model.load_state_dict(torch.load(
#         args.gcn_model_path, map_location=torch.device('cpu')))
#     gcn_model = gcn_model.to(device)

#     train_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args.train_input_file,
#         fit=True
#     )
#     dev_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args.dev_input_file,
#         fit=False,
#         all_edge_types=train_dataset.all_edge_types,
#         ohe=train_dataset.ohe
#     )

#     test_dataset = gcn.Logical_Fallacy_Dataset(
#         path_to_dataset=args.test_input_file,
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

#     train_predictions, train_confidence, all_train_sentences = gcn.do_predict_process(
#         model=gcn_model,
#         loader=train_data_loader
#     )
#     dev_predictions, dev_confidence, all_dev_sentences = gcn.do_predict_process(
#         model=gcn_model,
#         loader=dev_data_loader,
#     )
#     test_predictions, test_confidence, all_test_sentences = gcn.do_predict_process(
#         model=gcn_model,
#         loader=test_data_loader
#     )

#     del gcn_model

#     train_df = augment_records_with_similar_records(
#         data_df=train_df,
#         source_df=train_df,
#         num_cases=args.num_cases,
#         gcn_predictions=train_predictions,
#         gcn_all_confidences=train_confidence,
#         gcn_all_sentences=all_train_sentences
#     )
#     dev_df = augment_records_with_similar_records(
#         data_df=dev_df,
#         source_df=train_df,
#         num_cases=args.num_cases,
#         gcn_predictions=dev_predictions,
#         gcn_all_confidences=dev_confidence,
#         gcn_all_sentences=all_dev_sentences
#     )
#     test_df = augment_records_with_similar_records(
#         data_df=test_df,
#         source_df=train_df,
#         num_cases=args.num_cases,
#         gcn_predictions=test_predictions,
#         gcn_all_confidences=test_confidence,
#         gcn_all_sentences=all_test_sentences
#     )

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, \
    classification_report
import pandas as pd
import torch
import joblib
import wandb
import random
import numpy as np
from IPython import embed
from pathlib import Path
from transformers import TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import argparse

torch.manual_seed(77)
random.seed(77)
np.random.seed(77)

NUM_LABELS = 13


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
        if args.input_feature == "masked_articles":
            base_sentence = obj[1].sentence
        elif args.input_feature == "source_article":
            base_sentence = obj[0]
        else:
            raise NotImplementedError()
        # graph = obj[1].graph_nx
        augmented_sentences = []
        # if "wordnet" in augments:
        #     augmented_sentences.extend(
        #         list(get_wordnet_edges_in_sentences(
        #             graph,
        #             obj[1].label2word
        #         ))
        #     )
        # if "conceptnet" in augments:
        #     augmented_sentences.extend(
        #         list(get_conceptnet_edges_in_sentences(
        #             graph,
        #             obj[1].label2word
        #         ))
        #     )
        sentences.append(
            "; ".join([base_sentence, *augmented_sentences])
        )
        labels.append(obj[2])

    return pd.DataFrame({
        args.input_feature: sentences,
        "updated_label": labels
    })


def do_train_process(args):

    train_df = read_csv_from_amr(args.train_input_file)
    dev_df = read_csv_from_amr(args.dev_input_file)
    test_df = read_csv_from_amr(args.test_input_file)


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
        "roberta-base", num_labels=NUM_LABELS, classifier_dropout = args.classifier_dropout)

    print('Model loaded!')

    training_args = TrainingArguments(
        do_eval=True,
        do_train=True,
        output_dir="./xlm_roberta_logical_fallacy_classification",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
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

    train_predictions = label_encoder.inverse_transform(np.argmax(trainer.predict(tokenized_dataset['train']).predictions, axis = -1))
    train_true_labels = label_encoder.inverse_transform(tokenized_dataset['train']['labels'])

    print('performance on train data')
    print(classification_report(
        y_pred = train_predictions, 
        y_true = train_true_labels
    ))

    eval_predictions = label_encoder.inverse_transform(np.argmax(trainer.predict(tokenized_dataset['eval']).predictions, axis = -1))
    eval_true_labels = label_encoder.inverse_transform(tokenized_dataset['eval']['labels'])

    print('performance on eval data')
    print(classification_report(
        y_pred = eval_predictions,
        y_true = eval_true_labels
    ))

    test_predictions = label_encoder.inverse_transform(np.argmax(trainer.predict(tokenized_dataset['test']).predictions, axis = -1))
    test_true_labels = label_encoder.inverse_transform(tokenized_dataset['test']['labels'])

    print('performance on test data')
    print(classification_report(
        y_pred = test_predictions, 
        y_true = test_true_labels
    ))






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
    parser.add_argument(
        '--batch_size', type = int, default = 8
    )

    parser.add_argument(
        '--learning_rate', type = float, default = 2e-5
    )

    parser.add_argument(
        '--num_epochs', type = int, default = 10
    )

    parser.add_argument(
        '--classifier_dropout', type = float, default = 0.3
    )

    parser.add_argument(
        '--weight_decay', type = float, default = 0.01
    )


    args = parser.parse_args()

    wandb.init(
        project="Logical Fallacy Main Classifier",
        entity='zhpinkman',
        config={
            **vars(args)
        }
    )

    do_train_process(args)


    