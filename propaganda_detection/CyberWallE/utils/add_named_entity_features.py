"""
Named entity features for the technique classification task.
Encodes the presence of different kinds of named entities in a given text
fragment (multiple binary features).
"""
from collections import Counter
import spacy


def search(nlp, text, features):
    results = {f: '0' for f in features}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in features:
            results[ent.label_] = '1'
    # Sometimes multi-word units containing a country name or nationality are
    # classified as 'ORG' instead of 'GPE'/'NORP'
    # -> Check individual tokens (out of context)
    if len(doc) > 1 and '0' in results.values():
        for tok in doc:
            for ent in nlp(tok.text).ents:
                if ent.label_ in features:
                    results[ent.label_] = '1'
    return results


def annotate_file_ner(nlp, features, in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip())
        for feature in features:
            f.write('\t' + feature)
        f.write('\n')
        for idx, line in enumerate(lines[1:]):
            f.write(line.strip())
            text = line.split('\t')[4]
            results = search(nlp, text, features)
            for feature in features:
                f.write('\t' + results[feature])
            f.write('\n')
            if idx % 250 == 0:
                print(idx, text, results)


def search_all(nlp, text):
    results = set()
    doc = nlp(text)
    for ent in doc.ents:
        results.add(ent.label_)
    # Sometimes multi-word units containing a country name or nationality are
    # classified as 'ORG' instead of 'GPE'/'NORP'
    # -> Check individual tokens (out of context)
    if len(doc) > 1:
        for tok in doc:
            for ent in nlp(tok.text).ents:
                results.add(ent.label_)
    return results


def match_all_ne(nlp, in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    matches = []

    for idx, line in enumerate(lines[1:]):
        text = line.split('\t')[4]
        results = search_all(nlp, text)
        matches.extend(results)
        if idx % 250 == 0:
            print(idx, text, results)

    print(sorted(dict(Counter(matches)).items(), key=lambda item: item[1],
                 reverse=True))


nlp = spacy.load('en_core_web_sm')
# match_all_ne(nlp, '../data/tc-train.tsv')
# [('ORG', 729), ('NORP', 573), ('GPE', 522), ('PERSON', 507),
#  ('CARDINAL', 287), ('DATE', 163), ('ORDINAL', 69), ('LOC', 50),
#  ('WORK_OF_ART', 38), ('EVENT', 25), ('LAW', 19), ('TIME', 15),
#  ('MONEY', 13), ('PERCENT', 12), ('FAC', 11), ('PRODUCT', 7),
#  ('LANGUAGE', 5), ('QUANTITY', 3)]

features = ['ORG',
            'NORP',  # Nationalities or religious or political groups
            'GPE',   # Geopolitical entities: Countries, cities, states
            'PERSON', 'CARDINAL', 'DATE'
            ]
annotate_file_ner(nlp, features, '../data/tc-train.tsv')
annotate_file_ner(nlp, features, '../data/tc-dev.tsv')
annotate_file_ner(nlp, features, '../data/tc-test.tsv')
