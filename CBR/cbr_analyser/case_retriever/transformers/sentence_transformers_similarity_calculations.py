import joblib
from sentence_transformers import SentenceTransformer, util
from IPython import embed

model = SentenceTransformer('all-MiniLM-L6-v2')


sentences_with_amr_objects = joblib.load(
    f"tmp/masked_sentences_with_AMR_container_objects_train.joblib")
train_sentences = [obj[1].sentence.strip()
                   for obj in sentences_with_amr_objects]

train_sentences_embeddings = model.encode(
    train_sentences, convert_to_tensor=True)

for split in ["train", "dev", "test"]:
    sentences_with_amr_objects = joblib.load(
        f"tmp/masked_sentences_with_AMR_container_objects_{split}.joblib")
    all_sentences = [obj[1].sentence.strip()
                     for obj in sentences_with_amr_objects]

    all_sentences_embeddings = model.encode(
        all_sentences, convert_to_tensor=True)

    similarities = util.cos_sim(
        all_sentences_embeddings, train_sentences_embeddings)

    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))

    joblib.dump(similarities_dict,
                f"tmp/sentence_transformers_similarities_{split}.joblib")
