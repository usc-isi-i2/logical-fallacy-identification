import argparse

import joblib
from IPython import embed
from sentence_transformers import SentenceTransformer, util


def get_source_feature_from_amr_objects(sentences_with_amr_objects, source_feature):
    if source_feature == "source":
        return [obj[0].strip()
                for obj in sentences_with_amr_objects]
    elif source_feature == "masked":
        return [obj[1].sentence.strip()
                for obj in sentences_with_amr_objects]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_feature', choices=['masked', 'source'], type=str)
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--output_file', type=str)
    
    args = parser.parse_args()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences_with_amr_objects = joblib.load(args.source_file)
    train_sentences = get_source_feature_from_amr_objects(sentences_with_amr_objects, args.source_feature)
    
    sentences_with_amr_objects = joblib.load(args.target_file)
    all_sentences = get_source_feature_from_amr_objects(sentences_with_amr_objects, args.source_feature)

    train_sentences_embeddings = model.encode(
        train_sentences, convert_to_tensor=True)
    
    all_sentences_embeddings = model.encode(
            all_sentences, convert_to_tensor=True)
    
    similarities = util.cos_sim(
            all_sentences_embeddings, train_sentences_embeddings)
    
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(train_sentences, row.tolist()))
    
    joblib.dump(similarities_dict, args.output_file)
    

    