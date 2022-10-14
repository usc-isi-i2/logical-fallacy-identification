from simcse import SimCSE
import joblib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=['masked', 'source'], type=str)
parser.add_argument('--output_prefix', type=str)
args = parser.parse_args()

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

sentences_with_amr_objects = joblib.load(
    f"tmp/masked_sentences_with_AMR_container_objects_train.joblib")

if args.source == "source":
    train_sentences = [obj[0].strip()
                       for obj in sentences_with_amr_objects]
elif args.source == "masked":
    train_sentences = [obj[1].sentence.strip()
                       for obj in sentences_with_amr_objects]

for split in ["train", "dev", "test"]:
    sentences_with_amr_objects = joblib.load(
        f"tmp/masked_sentences_with_AMR_container_objects_{split}.joblib")
    if args.source == "source":
        all_sentences = [obj[0].strip()
                         for obj in sentences_with_amr_objects]
    elif args.source == "masked":
        all_sentences = [obj[1].sentence.strip()
                         for obj in sentences_with_amr_objects]
    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, f"tmp/{args.output_prefix}_{split}.joblib")
