import argparse
from torch.optim import Adam
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (DataCollatorWithPadding,
                          Trainer, TrainingArguments, RobertaTokenizer, RobertaModel, RobertaPreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput


from cbr_analyser.case_retriever.retriever import (Retriever, SimCSE_Retriever)
import wandb
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def augment_with_similar_cases(df: pd.DataFrame, retriever: Retriever, config, sep_token, prefix: str, is_train: bool) -> pd.DataFrame:
    external_sentences = []
    augmented_sentences = []
    count_without_cases = 0
    for sentence in df["text"]:
        try:
            similar_sentences_with_similarities = retriever.retrieve_similar_cases(
                sentence, config.num_cases)
            similar_sentences = [
                s[0] for s in similar_sentences_with_similarities if s[1] > config.cbr_threshold]
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


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def do_train_process(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = RobertaTokenizer.from_pretrained(
            "cross-encoder/nli-roberta-base")

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]

        if config.cbr == True:
            print('using cbr')
            simcse_retriever = SimCSE_Retriever(
                config={'data_dir': config.data_dir, 'source_feature': 'masked_articles'})

            for df, is_train in zip([train_df, dev_df, test_df], [True, False, False]):
                df = augment_with_similar_cases(
                    df, simcse_retriever, config, tokenizer.sep_token, "simcse", is_train=is_train)

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

        tokenizer = RobertaTokenizer.from_pretrained(
            "cross-encoder/nli-roberta-base")
        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = RobertaForSequenceClassification.from_pretrained(
            "cross-encoder/nli-roberta-base", num_labels=len(list(label_encoder.classes_)), classifier_dropout=config.classifier_dropout, ignore_mismatched_sizes=True)

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

        trainer = CustomTrainer(
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
            "values": [15]
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
