### Pre- and post-processing + evaluation code

Subtask 1: Span identification:
- `MergeTrainAndDev` creates the final input dataset
- `proprocess_si` extracts labels and sentences from the newspaper articles
- `add_pos_tags`, `add_rhetorical_features` and `add_sentiment_features` annotate the input data with additional features
- `bert_layer_extractor` extracts token-level BERT embeddings
- `postprocess_spans` merges nearby spans
- `eval` computes scores for our predicted spans, also on a technique level

Subtask 2: Technique classification
- `proprocess_tc` extracts labels and text fragments from the newspaper articles and does some pre-processing of the *repetition* class
- `add_repetition_features`, `add_named_entity_features` and `add_rhetorical_features` annotate the input data with additional features
- `add_bow_matches` and `add_emotion_features` + `ibm_api_config` add additional features for the feature ablation study