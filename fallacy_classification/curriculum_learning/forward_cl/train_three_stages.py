import argparse
from torch import nn
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
import numpy as np
from datasets import load_metric
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import glob, shutil

def load_data(path, class_list, tokenizer_name="", num_classes=13):
    features = Features({'text': Value('string'), 'label': ClassLabel(num_classes, class_list)})
    dataset = load_dataset('csv', data_files={"test": path,}, 
                        delimiter=',', column_names=['text', 'label'], 
                        skiprows=1, features=features,
                        keep_in_memory=True)

    # Load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512 if "nli-bert" in tokenizer_name else None)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    test_dataset = tokenized_datasets["test"]

    return test_dataset

def main(model_name, directory, tokenizer_name, dataset_train, dataset_val, dataset_test, num_classes, class_list=None, use_label_encoder=True, problem_type=None, epochs=10, weights=None, run_num=1):
    
    features = Features({'text': Value('string'), 'label': ClassLabel(num_classes, class_list)})
    dataset = load_dataset('csv', data_files={"train": dataset_train,
                                            "val": dataset_val,
                                            "test": dataset_test,}, 
                        delimiter=',', column_names=['text', 'label'], 
                        skiprows=1, features=features,
                        keep_in_memory=True)

    # Load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512 if "nli-bert" in tokenizer_name else None)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    # Extract splits

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["val"]
    test_dataset = tokenized_datasets["test"]


    # Load model with new head
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True, problem_type=problem_type)

    # Define metrics

    metric = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = metric.compute(predictions=predictions, references=labels)["accuracy"]
        prec = precision.compute(predictions=predictions, references=labels, average="weighted")["precision"]
        rec = recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]
        f1w = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        f1micro = f1.compute(predictions=predictions, references=labels, average="micro")["f1"]
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1_weighted": f1w, "f1_micro":f1micro}


    # Define training arguments for trainer
    training_args = TrainingArguments(output_dir=directory, 
                                    do_train=True,
                                    do_eval=True, 
                                    evaluation_strategy="epoch", 
                                    per_device_train_batch_size= 32 if "deberta" not in tokenizer_name and "electra" not in tokenizer_name else 16,
                                    per_device_eval_batch_size=32 if "deberta" not in tokenizer_name and "electra" not in tokenizer_name else 16, 
                                    learning_rate=5e-5,
                                    logging_strategy="epoch",
                                    num_train_epochs=epochs, 
                                    save_strategy="epoch", 
                                    save_total_limit=1, 
                                    load_best_model_at_end=True,
                                    metric_for_best_model="eval_f1_weighted",
                                    # metric_for_best_model="eval_loss",
                                    # greater_is_better=True,
                                    lr_scheduler_type="cosine")



    class CustomTrainer(Trainer):
        def __init__(self, weights, **kwargs):
            super().__init__(**kwargs)
            self.weights = weights

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights, device="cuda") if self.weights else None)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss


    # Define trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        weights=weights
    )


    checkpoints = glob.glob(directory + "/checkpoint*")
    for c in checkpoints:
        shutil.rmtree(c)
    
    # Train
    trainer.train()
    with open(os.path.join(directory, f"{tokenizer_name.replace('/', '_') + str(run_num)}.log"), "w+") as f:
        f.write(json.dumps(trainer.predict(test_dataset).metrics))
    
    if num_classes == 13:
        class_list = ['faulty generalization', 'ad hominem', 'false causality', 'ad populum', 'circular reasoning', 'appeal to emotion', 'fallacy of logic', 'fallacy of relevance', 'intentional', 'false dilemma', 'fallacy of credibility', 'fallacy of extension','equivocation']
        test_dataset = load_data("curriculum_learning_training/datasets/fine_grained_test.csv", class_list, tokenizer_name=tokenizer_name, num_classes=len(class_list))
        y_pred = torch.tensor(trainer.predict(test_dataset).predictions, dtype=torch.float32).max(1).indices.numpy()
        print("Classification report:\n\n", classification_report(test_dataset["label"], y_pred, target_names=class_list, digits=4))
        print(confusion_matrix(test_dataset["label"], y_pred, labels=list(range(0, len(class_list), 1))))



if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--directory_big_bench")
    parser.add_argument("--directory_coarse")
    parser.add_argument("--directory_fine")
    parser.add_argument("--tokenizer")
    parser.add_argument("--num_runs", type=int)

    parser.add_argument("--class_list_big_bench", type=list, default=["negative", "positive"])
    parser.add_argument("--class_list_coarse", type=list, default=["fallacy of relevance", "fallacies of defective induction", "fallacies of presumption", "fallacy of ambiguity"])
    parser.add_argument("--class_list_fine", type=list, default=['faulty generalization', 'ad hominem', 'false causality', 'ad populum', 'circular reasoning', 'appeal to emotion', 'fallacy of logic', 'fallacy of relevance', 'intentional', 'false dilemma', 'fallacy of credibility', 'fallacy of extension','equivocation'])

    parser.add_argument("--epochs_big_bench", type=int, default=5)
    parser.add_argument("--epochs_coarse", type=int, default=8)
    parser.add_argument("--epoch_fine", type=int, default=10)

    parser.add_argument("--big_bench_train_dataset", type=str)
    parser.add_argument("--big_bench_val_dataset", type=str)
    parser.add_argument("--big_bench_test_dataset", type=str)
    
    parser.add_argument("--coarse_train_dataset", type=str)
    parser.add_argument("--coarse_val_dataset", type=str)
    parser.add_argument("--coarse_test_dataset", type=str)

    parser.add_argument("--fine_train_dataset", type=str)
    parser.add_argument("--fine_val_dataset", type=str)
    parser.add_argument("--fine_test_dataset", type=str)

    args = parser.parse_args()

    # Run training on Big Bench only once because we don't see too much variance between runs for this dataset 
    print("printing", "-"*100, "\n", args.big_bench_train_dataset)
    main(args.model_name, args.directory_big_bench, args.tokenizer, 
        args.big_bench_train_dataset,
        args.big_bench_val_dataset,
        args.big_bench_test_dataset,
        num_classes=len(args.class_list_big_bench), 
        class_list = args.class_list_big_bench,
        problem_type="single_label_classification", 
        epochs=args.epochs_big_bench)
    
    for n in range(args.num_runs):
        big_bench_best_model = sorted(os.listdir(args.directory_big_bench))[0]
        
        main(args.directory_big_bench + "/" + big_bench_best_model, 
            args.directory_coarse, 
            args.tokenizer, 
            args.coarse_train_dataset,
            args.coarse_val_dataset,
            args.coarse_test_dataset,
            num_classes=len(args.class_list_coarse), 
            class_list=args.class_list_coarse, 
            problem_type="single_label_classification", 
            epochs=args.epochs_coarse, 
            run_num=n) # weights=[0.60662359, 0.52887324, 2.80223881, 9.62820513])
        
        coarse_best_model = sorted(os.listdir(args.directory_coarse))[0]
        main(args.directory_coarse + "/" + coarse_best_model, 
            args.directory_fine, 
            args.tokenizer, 
            args.fine_train_dataset,
            args.fine_val_dataset,
            args.fine_test_dataset,
            num_classes=len(args.class_list_fine), 
            class_list = args.class_list_fine,
            run_num=n) # weights=[0.44586448, 0.63213675, 0.81741821, 0.90019474, 1.06142365, 1.09408284, 1.1754609 , 1.24763833, 1.26991758, 1.29300699, 1.32925953, 1.34179971, 3.6469428])
