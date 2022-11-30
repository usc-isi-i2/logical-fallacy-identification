from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
import torch
import argparse
import pandas as pd

def load_data(path, class_list, tokenizer_name="", num_classes=16):
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


def get_predictions(path, num_classes, test_dataset, problem_type="single_label_classification"):
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_classes, ignore_mismatched_sizes=True, problem_type=problem_type)
    testing_args = TrainingArguments(output_dir="./test", 
                                    do_train=True,
                                    do_eval=False, 
                                    evaluation_strategy="epoch", 
                                    per_device_train_batch_size= 16,
                                    per_device_eval_batch_size=16, 
                                    logging_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=testing_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
    )
    
    return torch.tensor(trainer.predict(test_dataset).predictions, dtype=torch.float32).max(1).indices.numpy()

def test_classwise(tokenizer_name, test_dataset, path_model, class_list, num_classes=16):
    y_pred = get_predictions(path_model, num_classes, test_dataset)
    print(test_dataset["label"])
    print("Classification report:\n\n", classification_report(test_dataset["label"], y_pred, target_names=class_list, digits=4))
    print(confusion_matrix(test_dataset["label"], y_pred, labels=list(range(0, 15, 1))))
    df = pd.DataFrame({"text": test_dataset["text"], "ground_truth": test_dataset["label"], "prediction": y_pred})
    df.to_csv("/cluster/raid/home/darshan.deshpande/curriculum_learning_training/" + f"evaluate_{tokenizer_name.replace('/', '-')}_monitorf1.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--tokenizer")
    args = parser.parse_args()

    class_list = ['fallacy of red herring', 'faulty generalization', 'ad hominem', 'false causality', 'circular reasoning', 'ad populum', 'fallacy of credibility', 'appeal to emotion', 'fallacy of logic', 'intentional', 'fallacy of extension', 'false dilemma', 'equivocation', 'prejudicial language', 'slothful induction', 'fallacy of slippery slope']
    test_dataset = load_data("/cluster/raid/home/darshan.deshpande/curriculum_learning_training/datasets/fine_grained_test_final.csv", class_list, tokenizer_name=args.tokenizer)
    test_classwise(args.tokenizer, test_dataset, args.model_path, class_list)
    