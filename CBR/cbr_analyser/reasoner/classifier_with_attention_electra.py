from torch.nn import MultiheadAttention
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datetime import datetime
import wandb
from cbr_analyser.case_retriever.retriever import (
    Retriever, SentenceTransformerRetriever, SimCSE_Retriever, Empathy_Retriever)
import argparse
import joblib
from torch.optim import Adam
from IPython import embed
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers.activations import get_activation
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (DataCollatorWithPadding,
                          Trainer, TrainingArguments, ElectraModel, ElectraPreTrainedModel, ElectraTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput
os.environ["WANDB_MODE"] = "dryrun"


bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]


class ElectraClassificationHead(nn.Module):
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
        # although BERT uses tanh here, it seems Electra authors used gelu here
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.attention = MultiheadAttention(
            self.electra.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.classifier = ElectraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_cbr: Optional[torch.LongTensor] = None,
        attention_mask_cbr: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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

        discriminator_hidden_states = self.electra(
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

        sequence_output = discriminator_hidden_states[0]

        discriminator_hidden_states_cbr = self.electra(
            input_ids_cbr,
            attention_mask=attention_mask_cbr,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output_cbr = discriminator_hidden_states_cbr[0]

        final_output, _ = self.attention(
            query=sequence_output,
            key=sequence_output_cbr,
            value=sequence_output_cbr
        )

        logits = self.classifier(final_output)

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
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


def augment_with_similar_cases(df: pd.DataFrame, retrievers: List[Retriever], config, sep_token, train_df) -> pd.DataFrame:
    external_sentences = []
    augmented_sentences = []
    all_cbr_labels = []
    for sentence in tqdm(df["text"], leave=False):
        all_similar_sentences = []
        all_cases_labels = []
        for retriever in retrievers:
            try:
                similar_sentences_with_similarities = retriever.retrieve_similar_cases(
                    sentence, train_df, config.num_cases)
                similar_sentences_labels = [
                    (s[0], s[2]) for s in similar_sentences_with_similarities if s[1] > config.cbr_threshold]

                similar_sentences = [s[0] for s in similar_sentences_labels]
                similar_labels = [s[1] for s in similar_sentences_labels]

                all_similar_sentences.append(similar_sentences)
                all_cases_labels.append(similar_labels)
            except Exception as e:
                print(e)
        result_sentence = sentence
        for similar_sentences in all_similar_sentences:
            result_sentence += f" {sep_token}{sep_token} {' '.join(similar_sentences)}"
        all_cbr_labels.append(all_cases_labels)
        external_sentences.append(sep_token.join(
            [' '.join(similar_sentences) for similar_sentences in all_similar_sentences]))
        augmented_sentences.append(result_sentence)

    df["text_cbr"] = augmented_sentences
    df["cbr"] = external_sentences
    df["cbr_labels"] = all_cbr_labels
    return df


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_ids_cbr=inputs["input_ids_cbr"],
            attention_mask_cbr=inputs["attention_mask_cbr"],
        )

        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def do_train_process(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = ElectraTokenizer.from_pretrained(
            "howey/electra-base-mnli")

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        if config.data_dir != 'data/bigbench':
            climate_df = pd.read_csv(os.path.join(
                config.data_dir, "climate_test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        if config.data_dir != 'data/bigbench':
            climate_df = climate_df[~climate_df["label"].isin(bad_classes)]

        print('using cbr')

        retrievers_to_use = []
        for retriever_str in config.retrievers:
            if retriever_str == 'simcse':
                simcse_retriever = SimCSE_Retriever(
                    config={'data_dir': config.data_dir}
                )
                retrievers_to_use.append(simcse_retriever)
            elif retriever_str == 'empathy':
                empathy_retriever = Empathy_Retriever(
                    config={'data_dir': config.data_dir}
                )
                retrievers_to_use.append(empathy_retriever)
            elif retriever_str.startswith('sentence-transformers'):
                sentence_transformers_retriever = SentenceTransformerRetriever(
                    config={
                        'data_dir': config.data_dir,
                        'model_checkpoint': retriever_str
                    }
                )
                retrievers_to_use.append(sentence_transformers_retriever)

        dfs_to_process = [train_df, dev_df, test_df] if config.data_dir == 'data/bigbench' else [
            train_df, dev_df, test_df, climate_df]
        for df in dfs_to_process:
            df = augment_with_similar_cases(
                df, retrievers_to_use, config, tokenizer.sep_token, train_df
            )
        try:
            del retrievers_to_use
            del simcse_retriever
            del empathy_retriever
            del coarse_retriever
        except:
            pass

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['label'])

        train_df['label'] = label_encoder.transform(
            train_df['label'])
        dev_df['label'] = label_encoder.transform(dev_df['label'])
        test_df['label'] = label_encoder.transform(
            test_df['label'])
        if config.data_dir != 'data/bigbench':
            climate_df['label'] = label_encoder.transform(
                climate_df['label'])

        if config.data_dir == 'data/bigbench':
            dataset = DatasetDict({
                'train': Dataset.from_pandas(train_df),
                'eval': Dataset.from_pandas(dev_df),
                'test': Dataset.from_pandas(test_df),
            })
        else:
            dataset = DatasetDict({
                'train': Dataset.from_pandas(train_df),
                'eval': Dataset.from_pandas(dev_df),
                'test': Dataset.from_pandas(test_df),
                'climate': Dataset.from_pandas(climate_df)
            })

        def process(batch):
            inputs = tokenizer(
                batch["text"], truncation=True, padding='max_length')
            inputs_cbr = tokenizer(
                batch["text_cbr"], truncation=True, padding='max_length')
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'input_ids_cbr': inputs_cbr['input_ids'],
                'attention_mask_cbr': inputs_cbr['attention_mask'],
                'labels': batch['label']
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        model = ElectraForSequenceClassification.from_pretrained(
            "howey/electra-base-mnli", num_labels=len(list(label_encoder.classes_)), classifier_dropout=config.classifier_dropout, ignore_mismatched_sizes=True)

        print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"./cbr_electra_logical_fallacy_classification_{config.data_dir.replace('/', '_')}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy='steps',
            # report_to="wandb"
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
            compute_metrics=compute_metrics
        )

        print('Start the training ...')
        trainer.train()

        predictions = trainer.predict(tokenized_dataset['test'])
        if config.data_dir != 'data/bigbench':
            predictions_climate = trainer.predict(tokenized_dataset['climate'])

        # run_name = wandb.run.name
        outputs_dict = {}
        outputs_dict['note'] = 'best_hps_final_best_ps_electra'
        outputs_dict['label_encoder'] = label_encoder
        outputs_dict["meta"] = dict(config)
        # outputs_dict['run_name'] = run_name
        outputs_dict['predictions'] = predictions._asdict()
        if config.data_dir != 'data/bigbench':
            outputs_dict['predictions_climate'] = predictions_climate._asdict()
        outputs_dict['text'] = test_df['text'].tolist()

        outputs_dict['cbr_labels'] = test_df['cbr_labels'].tolist()
        outputs_dict['cbr'] = test_df['cbr'].tolist()

        now = datetime.today().isoformat()
        file_name = os.path.join(
            config.predictions_dir,
            f"outputs_dict__{now}.joblib"
        )
        print(file_name)
        joblib.dump(outputs_dict, file_name)
        print(predictions)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Classification Model for Logical Fallacy Detection and having a baseline')

    parser.add_argument(
        '--data_dir', help="Train input file path", type=str, default="data/new_finegrained"
    )
    parser.add_argument('--predictions_dir', help="Predictions output file path",
                        default="cache/predictions/all", type=str)

    parser.add_argument(
        '--checkpoint', help="Checkpoint namespace", type=str, default="simcse")

    parser.add_argument(
        '--num_cases', help="Number of cases in CBR", type=int, default=2)

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
        'retrievers': {
            "values": [
                [args.checkpoint]
                # ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'],
                # ['sentence-transformers/paraphrase-MiniLM-L6-v2'],
                # ['sentence-transformers/all-MiniLM-L12-v2'],
                # ['sentence-transformers/all-MiniLM-L6-v2'],
                # ['simcse'],
                # ['empathy'],
                # ["simcse", "empathy"],
                # ["simcse"],
                # ["empathy"]
            ]
        },
        'num_cases': {
            # "values": [4] if args.data_dir == "data/new_finegrained" else [1] if args.data_dir == "data/finegrained" else [1] if args.data_dir == "data/coarsegrained" else [3]
            # "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            "values": [args.num_cases]
        },
        'cbr_threshold': {
            "values": [-1e7, 0.5, 0.8]
            # "values": [0.5]
            # "values": [-10000000] if args.data_dir == "data/new_finegrained" else [-10000000] if args.data_dir == "data/finegrained" else [-10000000] if args.data_dir == "data/coarsegrained" else [0.5]
        },
        'data_dir': {
            "values": [args.data_dir]
        },
        'predictions_dir': {
            "values": [args.predictions_dir]
        },
        'batch_size': {
            "values": [16]
        },
        'learning_rate': {
            # 'values': [8.447927580802138e-05]
            'distribution': 'uniform',
            'min': 1e-5,
            'max': 1e-4
            # 'min': 3e-5 if args.data_dir == "data/finegrained" else 1e-5,
            # 'max': 6e-5 if args.data_dir == "data/finegrained" else 1e-4,
            # "values": [3.120210415844665e-05] if args.data_dir == "data/new_finegrained" else [7.484147412800621e-05] if args.data_dir == "data/finegrained" else [7.484147412800621e-05] if args.data_dir == "data/coarsegrained" else [5.393991227358502e-06]
        },
        "num_epochs": {
            "values": [10]
        },
        "classifier_dropout": {
            "values": [0.1, 0.3, 0.8]
            # 'values': [0.1]
            # "values": [0.8] if args.data_dir == "data/new_finegrained" else [0.3] if args.data_dir == "data/finegrained" else [0.3] if args.data_dir == "data/coarsegrained" else [0.1]
        },
        'weight_decay': {
            # 'values': [0.04962960561110768]
            'distribution': 'uniform',
            'min': 1e-4,
            'max': 1e-1
            # "values": [0.07600643653465429] if args.data_dir == "data/new_finegrained" else [0.00984762513370293] if args.data_dir == "data/finegrained" else [0.00984762513370293] if args.data_dir == "data/coarsegrained" else [0.022507698737927326]
        },
    }

    # checkpoints = [
    #     'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    #     'sentence-transformers/paraphrase-MiniLM-L6-v2',
    #     'sentence-transformers/all-MiniLM-L12-v2',
    #     'sentence-transformers/all-MiniLM-L6-v2',
    #     'simcse',
    #     'empathy',
    # ]

    # for checkpoint in checkpoints:
    # for i in range(6, 10):
    #     new_dict = {}
    #     for k, v in parameters_dict.items():
    #         new_dict[k] = v['values'][0]
    #     # new_dict['retrievers'] = [checkpoint]

    #     new_dict['num_cases'] = i + 1

    #     print(new_dict)
    #     new_dict = AttributeDict(new_dict)

    #     do_train_process(new_dict)

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(
        sweep_config, project="Baseline Finder with CBR and different retrievers")
    wandb.agent(sweep_id, do_train_process, count=3)
