"""
Part-of-speech tags of the tokens used as input to the span identification
task. One-hot encoding (saved as 15 binary features).
"""

import numpy as np
import spacy


def parse_input_file(infile, outfile):
    nlp = spacy.load("en_core_web_sm")
    pos_tags = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                'PART', 'PRON', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'X']
    tags2onehot = {}
    M = len(pos_tags)
    for tag in pos_tags:
        enc = np.eye(N=1, M=M, k=pos_tags.index(tag), dtype=int).tolist()[0]
        tags2onehot[tag] = [str(x) for x in enc]

    with open(infile, encoding='utf8') as f_in:
        lines = f_in.readlines()
        lines.append('eof\teof\teof\teof\teof\teof\n')

    rows = []
    tokens = []
    prev_article = ''
    first_line = True
    with open(outfile, 'w', encoding='utf8') as f_out:
        for line in lines:
            # Comments + header
            if line.startswith('#'):
                f_out.write(line)
                continue
            if first_line:
                f_out.write('# POS tags = spacy\n')
                first_line = False
                labels = line.strip().split('\t')
                try:
                    doc_idx = labels.index('document_id')
                except ValueError:
                    doc_idx = 0
                try:
                    word_idx = labels.index('token')
                except ValueError:
                    word_idx = 4
                labels += pos_tags
                f_out.write('\t'.join(labels))
                f_out.write('\n')
                continue

            line = line[:-1]  # Remove \n
            fields = line.split('\t')
            article = fields[doc_idx]
            word = fields[word_idx]

            if article != prev_article:
                doc = nlp(' '.join(tokens))
                for tok, row in zip(doc, rows):
                    f_out.write(row)
                    try:
                        f_out.write('\t')
                        f_out.write('\t'.join(tags2onehot[tok.pos_]))
                    except KeyError:  # unknown tag
                        for _ in pos_tags:
                            f_out.write('\t0')
                    f_out.write('\n')
                tokens = []
                rows = []
            tokens.append(word)
            rows.append(line)
            prev_article = article


parse_input_file('../data/train-improved-sentiwordnet-arguingfullindiv.tsv',
                 '../data/train-improved-sentiwordnet-arguingfullindiv-pos.tsv')
parse_input_file('../data/dev-improved-sentiwordnet-arguingfullindiv.tsv',
                 '../data/dev-improved-sentiwordnet-arguingfullindiv-pos.tsv')
parse_input_file('../data/test-improved-sentiwordnet-arguingfullindiv.tsv',
                 '../data/test-improved-sentiwordnet-arguingfullindiv-pos.tsv')
