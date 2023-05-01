import os
import pandas as pd
import torch
import wandb
from transformers import BartTokenizer, RobertaTokenizer, ElectraTokenizer

from args import args, bad_classes, datasets_config
from preprocess import CustomNonBinaryClassDataset
from utils import evaluate, print_logs
from models import ProtoTEx, ProtoTEx_roberta, ProtoTEx_electra


def main():
    ## preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    ## https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = BartTokenizer.from_pretrained("ModelTC/bart-base-mnli")
    elif args.architecture == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    elif args.architecture == "Electra":
        tokenizer = ElectraTokenizer.from_pretrained("howey/electra-base-mnli")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    test_df = test_df[
        ~test_df[datasets_config[args.data_dir]["features"]["label"]].isin(bad_classes)
    ]
    test_sentences = test_df[
        datasets_config[args.data_dir]["features"]["text"]
    ].tolist()
    test_labels = test_df[datasets_config[args.data_dir]["features"]["label"]].tolist()
    test_dataset = CustomNonBinaryClassDataset(
        sentences=test_sentences, labels=test_labels, tokenizer=tokenizer
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn
    )

    # Initialize wandb
    wandb.init(
        # Set the project where this run will be logged
        project=args.project, 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.experiment, 
        # Track hyperparameters and run metadata
        config={
            "num_pos_prototypes": args.num_pos_prototypes,
            "num_neg_prototypes": args.num_prototypes - args.num_pos_prototypes,
            "none_class": args.none_class,
            "augmentation": args.augmentation,
            "nli_intialization": args.nli_intialization,
            "curriculum": args.curriculum,
            "architecture": args.architecture,
        }
    )

    if args.model == "ProtoTEx":
        print(
            "ProtoTEx best model: {0}, {1}".format(
                args.num_prototypes, args.num_pos_prototypes
            )
        )
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx(
                num_prototypes=args.num_prototypes,
                num_pos_prototypes=args.num_pos_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
            ).cuda()
        elif args.architecture == "RoBERTa":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx_roberta(
                num_prototypes=args.num_prototypes,
                num_pos_prototypes=args.num_pos_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
            ).cuda()
        elif args.architecture == "Electra":
            print(f"Using backone: {args.architecture}")
            model = ProtoTEx_electra(
                num_prototypes=args.num_prototypes,
                num_pos_prototypes=args.num_pos_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
            ).cuda()
        else:
            print(f"Invalid backbone architecture: {args.architecture}")

        print(f"Loading model checkpoint: {args.model_checkpoint}")
        pretrained_dict = torch.load(args.model_checkpoint)
        # Fiter out unneccessary keys
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping weights for: {k}")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        total_loss, mac_prec, mac_recall, mac_f1_score, accuracy = evaluate(
            test_dl, model_new=model
        )
        logs_path = "Logs/" + args.modelname
        f = open(logs_path, "w")
        f.writelines([""])
        f.close()
        print_logs(
            logs_path,
            "TEST SCORES",
            0,
            total_loss,
            mac_prec,
            mac_recall,
            mac_f1_score,
            accuracy,
        )
        wandb.log(
            {
                "Test epoch": 0,
                "Test loss": total_loss,
                "Test Precision": mac_prec,
                "Test Recall": mac_recall, 
                "Test Accuracy": accuracy,
                "Test F1 score": mac_f1_score,
            }
        )


if __name__ == "__main__":
    main()
