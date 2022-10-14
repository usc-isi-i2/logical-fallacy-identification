import argparse

import joblib
from simcse import SimCSE


def get_source_feature_from_amr_objects(sentences_with_amr_objects, source_feature):
    if source_feature == "source":
        return [obj[0].strip()
                for obj in sentences_with_amr_objects]
    elif source_feature == "masked":
        return [obj[1].sentence.strip()
                for obj in sentences_with_amr_objects]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_feature',
                        choices=['masked', 'source'], type=str)
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    sentences_with_amr_objects = joblib.load(args.source_file)

    train_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, args.source_feature)

    sentences_with_amr_objects = joblib.load(args.target_file)
    all_sentences = get_source_feature_from_amr_objects(
        sentences_with_amr_objects, args.source_feature)
    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, args.output_file)
