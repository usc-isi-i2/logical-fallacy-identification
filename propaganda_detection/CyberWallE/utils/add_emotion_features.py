"""
Emotion features for the technique classifation task. Used for preliminary
experiments (-> feature ablation), but not in the final model.
Five binary features encoding the fear, sadnass, joy, anger and disgust scores
for a given text fragment.
The scores are obtained with IBM's Natural Language Understanding tool:
https://cloud.ibm.com/catalog/services/natural-language-understanding
"""
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core.api_exception import ApiException
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
import ibm_api_config


def get_nlu():
    nlu = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        authenticator=IAMAuthenticator(ibm_api_config.API_KEY)
    )
    nlu.set_service_url(ibm_api_config.SERVICE_URL)
    return nlu


def annotate_file(nlu, in_file, offset=0, n_steps=10000):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tfear\tsadness\tjoy\tanger\tdisgust\n')
        for idx, line in enumerate(lines[1:]):
            if idx < offset or idx >= offset + n_steps:
                f.write(line)
                continue
            text = line.split('\t')[4]
            try:
                scores = nlu.analyze(text=text, language='en',
                                       features=Features(emotion=EmotionOptions()))
                scores = scores.get_result()['emotion']['document']['emotion']
                f.write(line.strip())
                for e in ['fear', 'sadness', 'joy', 'anger', 'disgust']:
                    f.write('\t' + str(scores[e]))
                f.write('\n')
                if idx % 250 == 0:
                    print(idx, text, scores)
            except ApiException:
                print('ApiException:', ApiException)
                print('Successfully added emotion scores to instances ' +
                      str(offset) + '-' + str(idx - 1) + '.')
                print('Will copy the lines for this instance and the remaining'
                      ' ones, so you can rerun this method on the same file'
                      ' after fixing the bug.')


nlu = get_nlu()
annotate_file(nlu, '../data/tc-train.tsv')
annotate_file(nlu, '../data/tc-dev.tsv')
annotate_file(nlu, '../data/tc-test.tsv')
