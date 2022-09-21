"""
Bag-of-words matches for the technique classifation task. Used for preliminary
experiments (-> feature ablation), but not in the final model.

Binary features encoding whether at least one word of a certain category
occurs in a given text fragment.

- 'America': the word 'America' + terms relating to US citizens
- 'America-simple': 'America' as token or subtoken (not used)
- 'Reductio': words commonly used in 'reductio ad hitlerum' fragments
"""
import re


REGEX = re.compile('\\bamerica\\b')
reductio_ad_hitlerum = ['hitler', 'nazi', 'fascis',  # -t,m
                        'stalin'  # also commonly used in reductio fragments
                        ]


def find_america(text):
    if 'america' not in text:
        return '0'
    for phrase in ['american people', 'americans', 'american citizen']:
        if phrase in text:
            return '1'
    if REGEX.search(text):
        return '1'
    return '0'


def annotate_file_america(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tamerica\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            f.write(find_america(text) + '\n')


def annotate_file_america_simple(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tamerica_simple\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            if 'america' in text:
                f.write('1\n')
            else:
                f.write('0\n')


def match(text, words):
    for word in words:
        if word in text:
            return '1'
    return '0'


def annotate_file(words, in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\treductio\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4].lower()
            f.write(match(text, words))
            f.write('\n')


annotate_file_america('../data/tc-train.tsv')
annotate_file_america('../data/tc-dev.tsv')
annotate_file_america('../data/tc-test.tsv')

annotate_file_america_simple('../data/tc-train.tsv')
annotate_file_america_simple('../data/tc-dev.tsv')
annotate_file_america_simple('../data/tc-test.tsv')

annotate_file(reductio_ad_hitlerum, '../data/tc-train.tsv')
annotate_file(reductio_ad_hitlerum, '../data/tc-dev.tsv')
annotate_file(reductio_ad_hitlerum, '../data/tc-test.tsv')
