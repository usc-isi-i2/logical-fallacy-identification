from simcse import SimCSE
import joblib


model = SimCSE("princeton-nlp/sup-simcse-roberta-large")


for split in ["train", "dev", "test"]:
    sentences_with_amr_objects = joblib.load(f"tmp/masked_sentences_with_AMR_container_objects_{split}.joblib")
    all_sentences = [obj[1].sentence for obj in sentences_with_amr_objects]
    similarities = model.similarity(all_sentences, all_sentences)
    



