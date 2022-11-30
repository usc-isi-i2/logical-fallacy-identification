import argparse
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
import random
import re
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import math
from torch.nn import functional as F
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from IPython import embed
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)
from transformers import RobertaTokenizer, RobertaModel
import cbr_analyser.consts as consts
import wandb
from cbr_analyser.case_retriever.retriever import (Empathy_Retriever,
                                                   Retriever, SimCSE_Retriever)
from torch import nn

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../amr/"))

# torch.manual_seed(77)
# random.seed(77)
# np.random.seed(77)

ROBERTA_HIDDEN_SIZE = 768

def attention(q, k, v, d_k, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class Attn_Network(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_Network, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.d_k = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, v, k):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, 1, self.d_model)
        q = self.q_linear(q).view(bs, -1, 1, self.d_model)
        v = self.v_linear(v).view(bs, -1, 1, self.d_model)

        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        
        scores = attention(q, k, v, self.d_k, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        

class CBR_Classifier(nn.Module):
    def __init__(self, num_labels, encoder_dropout_rate: float =0.5, attn_dropout_rate: float =0.4, last_layer_dropout: float =0.2):
        super().__init__()
        self.num_labels = num_labels
        
        self.normal_encoder = RobertaModel.from_pretrained("cross-encoder/nli-roberta-base")
        self.dropout1 = nn.Dropout(encoder_dropout_rate)
        
        self.simcse_encoder = RobertaModel.from_pretrained("cross-encoder/nli-roberta-base")
        self.dropout2 = nn.Dropout(encoder_dropout_rate)
        
        self.empathy_encoder = RobertaModel.from_pretrained("cross-encoder/nli-roberta-base")
        self.dropout3 = nn.Dropout(encoder_dropout_rate)
        
        self.f1 = nn.Linear(ROBERTA_HIDDEN_SIZE, 128)
        self.f2 = nn.Linear(ROBERTA_HIDDEN_SIZE, 128)
        self.f3 = nn.Linear(ROBERTA_HIDDEN_SIZE, 128)
        
        self.batch_norm1 = nn.BatchNorm1d(128 * 3)
        # self.batch_norm2 = nn.BatchNorm1d(ROBERTA_HIDDEN_SIZE)
        self.dropout_att = nn.Dropout(attn_dropout_rate)
        
        self.attn = Attn_Network(128)
        
        self.f4 = nn.Linear(128 * 3, 64)
        self.dropout4 = nn.Dropout(last_layer_dropout)
        self.f5 = nn.Linear(64, self.num_labels)
        

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3):
        bs = input_ids1.shape[0]
        
        x1 = self.normal_encoder(input_ids1, attention_mask=attention_mask1).pooler_output
        x1 = nn.ReLU()(x1)
        x1 = self.dropout1(x1)
        x1 = self.f1(x1)
        
        
        x2 = self.simcse_encoder(input_ids2, attention_mask=attention_mask2).pooler_output
        x2 = nn.ReLU()(x2)
        x2 = self.dropout2(x2)
        x2 = self.f2(x2)
        
        
        x3 = self.empathy_encoder(input_ids3, attention_mask=attention_mask3).pooler_output
        x3 = nn.ReLU()(x3)
        x3 = self.dropout3(x3)
        x3 = self.f3(x3)
        
        
        x_combined = torch.cat([x1, x2, x3], dim=-1)
        
        x_combined_norm = self.batch_norm1(x_combined)
        
        x_combined = x_combined + self.dropout_att(
            self.attn(
                x_combined_norm.view(bs, 3, 128),
                x_combined_norm.view(bs, 3, 128), 
                x_combined_norm.view(bs, 3, 128)
            ).view(bs, 3 * 128)
        )
        
        
        # x_combined = self.batch_norm2(x2)
        x_combined = self.f4(x_combined)
        x_combined = nn.ReLU()(x_combined)
        x_combined = self.dropout4(x_combined)
        
        x_combined = self.f5(x_combined)
        
        return x_combined 



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

def augment_with_similar_cases(df: pd.DataFrame, retriever: Retriever, args: Dict[str, Any], sep_token, prefix: str, is_train: bool) -> pd.DataFrame:    
    external_sentences = []
    augmented_sentences = []
    count_without_cases = 0
    for sentence in df["text"]:
        try:
            similar_sentences_with_similarities = retriever.retrieve_similar_cases(sentence, args["num_cases"])
            similar_sentences = [s[0] for s in similar_sentences_with_similarities if s[1] > args['cbr_threshold']]
            result_sentence = f"{sentence} {sep_token}{sep_token} {' '.join(similar_sentences)}"
            external_sentences.append('</sep>'.join(similar_sentences))
            augmented_sentences.append(result_sentence)
            # all_labels.append(label)
        except Exception as e:
            print(e)
            count_without_cases += 1
            result_sentence = sentence
            external_sentences.append('')
            augmented_sentences.append(result_sentence)
            # all_labels.append(label)

            
    df[f"text_{prefix}"] = augmented_sentences
    df[f'cbr_{prefix}'] = external_sentences
    return df

    


    
def train(model, data_loader, data_loader_simcse, data_loader_empathy, optimizer, loss_fn, logger):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0

    for index, (data, data_simcse, data_empathy) in enumerate(zip(data_loader, data_loader_simcse, data_loader_empathy)):
        input_ids1 = data['input_ids'].to(device)
        attention_mask1 = data['attention_mask'].to(device)
        
        input_ids2 = data_simcse['input_ids'].to(device)
        attention_mask2 = data_simcse['attention_mask'].to(device)
        
        input_ids3 = data_empathy['input_ids'].to(device)
        attention_mask3 = data_empathy['attention_mask'].to(device)

        labels = data['labels'].to(device)

        optimizer.zero_grad()


        outputs = model(
                input_ids1 = input_ids1, 
                attention_mask1 = attention_mask1, 
                input_ids2=input_ids2, 
                attention_mask2=attention_mask2,
                input_ids3 = input_ids3, 
                attention_mask3 = attention_mask3
            )
        
        input_ids1 = input_ids1.detach().cpu()
        attention_mask1  = attention_mask1.detach().cpu()
        input_ids2 = input_ids2.detach().cpu()
        attention_mask2 = attention_mask2.detach().cpu()
        input_ids3 = input_ids3.detach().cpu()
        attention_mask3 = attention_mask3.detach().cpu()
        
        loss = loss_fn(outputs, labels)
        
        outputs = outputs.detach().cpu()
        
        total_loss += loss.item()

        if (index + 1) % 100 == 0:
            print(f'[{index + 1:3d}] loss: {total_loss / 100:.3f}')
            total_loss = 0


        loss.backward()
        optimizer.step()

        logger.update(1)    

def evaluate(model, data_loader, data_loader_simcse, data_loader_empathy, loss_fn, data_type, label_encoder_inverse):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss = 0
    total_size = 0
    
    all_predictions = []
    all_labels = []


    for data, data_simcse, data_empathy in zip(data_loader, data_loader_simcse, data_loader_empathy):
        input_ids1 = data['input_ids'].to(device)
        attention_mask1 = data['attention_mask'].to(device)
        
        input_ids2 = data_simcse['input_ids'].to(device)
        attention_mask2 = data_simcse['attention_mask'].to(device)
        
        input_ids3 = data_empathy['input_ids'].to(device)
        attention_mask3 = data_empathy['attention_mask'].to(device)
        

        labels = data['labels'].to(device)
        total_size += labels.size(0)

        with torch.no_grad():
            outputs = model(
                input_ids1 = input_ids1, 
                attention_mask1 = attention_mask1, 
                input_ids2=input_ids2, 
                attention_mask2=attention_mask2,
                input_ids3 = input_ids3, 
                attention_mask3 = attention_mask3
            )
            
            input_ids1 = input_ids1.detach().cpu()
            attention_mask1  = attention_mask1.detach().cpu()
            input_ids2 = input_ids2.detach().cpu()
            attention_mask2 = attention_mask2.detach().cpu()
            input_ids3 = input_ids3.detach().cpu()
            attention_mask3 = attention_mask3.detach().cpu()

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim = -1)
            
            outputs = outputs.detach().cpu()
                        
            labels = labels.detach().cpu().numpy().tolist()
            predictions = predictions.detach().cpu().numpy().tolist()
            
            labels = list(map(label_encoder_inverse, labels))
            all_labels.extend(labels)
            predictions = list(map(label_encoder_inverse, predictions))
            all_predictions.extend(predictions)

    print(
        f'{data_type} Loss: {total_loss / total_size: .3f} \
        | {data_type} Accuracy: {accuracy_score(y_true = all_labels, y_pred = all_predictions) : .3f} \
        | {data_type} F1: {f1_score(y_true = all_labels, y_pred = all_predictions, average = "weighted") : .3f}'
    )
    return {
        'loss': (total_loss / total_size),
        'accuracy': accuracy_score(y_true=all_labels, y_pred= all_predictions),
        'precision': precision_score(y_true=all_labels, y_pred= all_predictions, average = "weighted"),
        'recall': recall_score(y_true=all_labels, y_pred= all_predictions, average = "weighted"),
        'f1': f1_score(y_true=all_labels, y_pred= all_predictions, average = "weighted"),
        'all_labels': all_labels,
        'all_predictions': all_predictions
    }
        

def do_train_process(config=None):
    print('Training process started')
    
    args = config
    print(args)
    
    
    roberta_tokenizer = RobertaTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    
    
    # train_df = read_csv_from_amr(input_file=args["train_input_file"], augments=args["augments"], source_feature=args["source_feature"])
    # dev_df = read_csv_from_amr(input_file=args["dev_input_file"], augments=args["augments"], source_feature=args["source_feature"])
    # test_df = read_csv_from_amr(input_file=args["test_input_file"], augments=args["augments"], source_feature=args["source_feature"])
    
    train_df = pd.read_csv(args["train_input_file"])
    dev_df = pd.read_csv(args["dev_input_file"])
    test_df = pd.read_csv(args["test_input_file"])
    
    
    train_df = train_df[~train_df["label"].isin(consts.bad_classes)]
    dev_df = dev_df[~dev_df["label"].isin(consts.bad_classes)]
    test_df = test_df[~test_df["label"].isin(consts.bad_classes)]

    if args["cbr"]:
        simcse_retriever = SimCSE_Retriever(config = args)
        empathy_retriever = Empathy_Retriever(config = args)
        
        
        for df, is_train in zip([train_df, dev_df, test_df], [True, False, False]):
            for retriever, prefix in zip([simcse_retriever, empathy_retriever], ["simcse", "empathy"]):
                df = augment_with_similar_cases(df, retriever, args, roberta_tokenizer.sep_token, prefix, is_train = is_train)
                
    # embed()

    label2index = consts.datasets_config[args.data_dir]["classes"]
    index2label = {v: k for k, v in label2index.items()}
    label_encoder = lambda x: label2index[x]
    label_encoder_inverse = lambda x: index2label[x]


    train_df['label'] = list(map(label_encoder, train_df['label']))
    dev_df['label'] = list(map(label_encoder, dev_df['label']))
    test_df['label'] = list(map(label_encoder, test_df['label']))

    # embed()
    # exit()

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'eval': Dataset.from_pandas(dev_df),
        'test': Dataset.from_pandas(test_df)
    })

    def process(batch):
        texts = batch["text"]
        inputs = roberta_tokenizer(texts, truncation=True)
        return {
            **inputs,
            'labels': batch['label']
        }
    def process_1(batch):
        texts = batch["text_simcse"]
        inputs = roberta_tokenizer(texts, truncation=True)
        return {
            **inputs,
            'labels': batch['label']
        }
    def process_2(batch):
        texts = batch["text_empathy"]
        inputs = roberta_tokenizer(texts, truncation=True)
        return {
            **inputs,
            'labels': batch['label']
        }


    tokenized_dataset = dataset.map(
        process, batched=True, remove_columns=dataset['train'].column_names)

    tokenized_dataset_simcse = dataset.map(
        process_1, batched=True, remove_columns=dataset['train'].column_names)
    
    tokenized_dataset_empathy = dataset.map(
        process_2, batched=True, remove_columns=dataset['train'].column_names)
    
    data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    
    train_data_loader = DataLoader(tokenized_dataset['train'], batch_size=args["batch_size"], shuffle=False, collate_fn=data_collator)
    dev_data_loader = DataLoader(tokenized_dataset['eval'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    test_data_loader = DataLoader(tokenized_dataset['test'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    
    simcse_train_data_loader = DataLoader(tokenized_dataset_simcse['train'], batch_size=args["batch_size"], shuffle=False, collate_fn=data_collator)
    simcse_dev_data_loader = DataLoader(tokenized_dataset_simcse['eval'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    simcse_test_data_loader = DataLoader(tokenized_dataset_simcse['test'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    
    empathy_train_data_loader = DataLoader(tokenized_dataset_empathy['train'], batch_size=args["batch_size"], shuffle=False, collate_fn=data_collator)
    empathy_dev_data_loader = DataLoader(tokenized_dataset_empathy['eval'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    empathy_test_data_loader = DataLoader(tokenized_dataset_empathy['test'], batch_size=args["batch_size"]*4, shuffle=False, collate_fn=data_collator)
    
    
    print("Data loaded")
    
    if 'encoder_dropout_rate' in args:
        model = CBR_Classifier(
            num_labels = len(consts.datasets_config[args.data_dir]["classes"]),
            encoder_dropout_rate = args['encoder_dropout_rate'], 
            attn_dropout_rate = args['attn_dropout_rate'], 
            last_layer_dropout = args['last_layer_dropout']
        )
    else:
        model = CBR_Classifier(num_labels = len(consts.datasets_config[args.data_dir]["classes"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    print('Model loaded!')


    loss_fn = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr = args['learning_rate'], eps = 1e-8, weight_decay = args['weight_decay'])
    
    
    evaluate(
        model = model, 
        data_loader = train_data_loader,
        data_loader_simcse=simcse_train_data_loader,
        data_loader_empathy=empathy_train_data_loader,
        loss_fn = loss_fn,
        data_type='Valid',
        label_encoder_inverse = label_encoder_inverse   
    )

    logging_steps = args["num_epochs"] * len(train_data_loader)
    logger = tqdm(range(logging_steps))

    for epoch in range(args["num_epochs"]):
        train(
            model = model, 
            data_loader = train_data_loader, 
            data_loader_simcse = simcse_train_data_loader, 
            data_loader_empathy = empathy_train_data_loader,
            optimizer = optimizer, 
            loss_fn = loss_fn, 
            logger = logger
        )
        train_metrics = evaluate(
            model = model, 
            data_loader = train_data_loader,
            data_loader_simcse=simcse_train_data_loader,
            data_loader_empathy=empathy_train_data_loader,
            loss_fn = loss_fn,
            data_type='Train',
            label_encoder_inverse = label_encoder_inverse
        )
        eval_metrics = evaluate(
            model = model, 
            data_loader = dev_data_loader,
            data_loader_simcse=simcse_dev_data_loader,
            data_loader_empathy=empathy_dev_data_loader,
            loss_fn = loss_fn,
            data_type='Valid',
            label_encoder_inverse = label_encoder_inverse
        )
        test_metrics = evaluate(
            model = model, 
            data_loader = test_data_loader,
            data_loader_simcse=simcse_test_data_loader,
            data_loader_empathy=empathy_test_data_loader,
            loss_fn = loss_fn,
            data_type='Test',
            label_encoder_inverse = label_encoder_inverse
        )
        yield {
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'valid_loss': eval_metrics['loss'],
                'valid_accuracy': eval_metrics['accuracy'],
                'valid_f1': eval_metrics['f1'],
                'valid_precision': eval_metrics['precision'],
                'valid_recall': eval_metrics['recall'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'epoch': epoch
            }
            
    if args["predictions_path"]:
        os.makedirs(args["predictions_path"], exist_ok=True)
        joblib.dump(
            {
                **test_metrics,
                'simcse_examples': dataset['test']['cbr_simcse'],
                'empathy_examples': dataset['test']['cbr_empathy']
            },
            os.path.join(
                args["predictions_path"],
                f"test_metrics_{args['run_name']}.joblib"
            )
        )
        joblib.dump(
            {
                **eval_metrics,
                'simcse_examples': dataset['eval']['cbr_simcse'],
                'empathy_examples': dataset['eval']['cbr_empathy']
            }
            ,
            os.path.join(
                args["predictions_path"],
                f"eval_metrics_{args['run_name']}.joblib"
            )
        )
    
        
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


    