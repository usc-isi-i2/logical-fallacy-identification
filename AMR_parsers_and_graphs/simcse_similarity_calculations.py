from simcse import SimCSE
import joblib


model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

sentences_with_amr_objects = joblib.load(
    f"tmp/masked_sentences_with_AMR_container_objects_train.joblib")
train_sentences = [obj[1].sentence.strip()
                   for obj in sentences_with_amr_objects]

for split in ["train", "dev", "test"]:
    sentences_with_amr_objects = joblib.load(
        f"tmp/masked_sentences_with_AMR_container_objects_{split}.joblib")
    all_sentences = [obj[1].sentence.strip()
                     for obj in sentences_with_amr_objects]
    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict, f"tmp/simcse_similarities_{split}.joblib")
