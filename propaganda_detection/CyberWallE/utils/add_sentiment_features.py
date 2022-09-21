"""
Sentiment features for the span identification task.
Encodes how positive and how negative a given token is.

Based on SentiWordNet by Esuli & Sebastiani (2006):
https://github.com/aesuli/sentiwordnet
https://github.com/aesuli/SentiWordNet/blob/master/papers/LREC06.pdf

Also creates sentiment features we briefly used in preliminary experiments:
- Span identification: single feature encoding how positive or negative a given
  token is (based on SentiWords by Gatti, Guerini & Turchi, 2016,
  https://hlt-nlp.fbk.eu/technologies/sentiwords)
- Technique classification: two features encoding the SentiWordNet scores of
  the most positive and negative tokens in a given text fragment.
"""
from spacy.lang.en import English


SENTIWORDS = '../data/sentiment/SentiWords_1.1.txt'
SENTIWORDNET = '../data/sentiment/SentiWordNet_3.0.0.txt'


def parse_sentiwordnet(lexicon_file):
    """
    Creates a dict(str -> (float, float)) from words to positive and negative
    scores. If a word contains several entries, the scores are averaged.
    """
    lex = dict()
    with open(lexicon_file, encoding='utf8') as f:
        for line in f:
            if line.startswith('#'):
                # comment
                continue
            fields = line.strip().split('\t')
            if len(fields) < 6:
                # last line
                continue
            # postag    id  score_pos   score_neg   word#sense word2#sense  def
            pos = float(fields[2])
            neg = float(fields[3])
            for word in fields[4].split():
                word = word.split('#')[0]
                try:
                    prev_pos, prev_neg, count = lex[word]
                    lex[word] = (prev_pos + pos, prev_neg + neg, count + 1)
                except KeyError:
                    lex[word] = (pos, neg, 1)

    for word in lex:
        pos, neg, count = lex[word]
        lex[word] = (pos / count, neg / count)

    return lex


def parse_sentiwords(lexicon_file):
    lex = dict()
    prev_word = ''
    score = 0
    n_entries = 0

    lines = []
    with open(lexicon_file, encoding='utf8') as f:
        lines = f.readlines()
        lines += ['end-of-file\t0']

    for line in lines:
        if line.startswith('#'):
            # comment
            continue
        fields = line.split('\t')
        # word#pos    value
        word = fields[0].split('#')[0]
        value = float(fields[1])
        if word == prev_word:
            score += value
            n_entries += 1
        else:
            if n_entries > 0:
                lex[word] = score / n_entries
            n_entries = 0
            score = 0
        prev_word = word

    return lex


def annotate_tokens(lex, infile, outfile, nlp, sentiwordnet):
    with open(infile, encoding='utf8') as f_in:
        with open(outfile, 'w', encoding='utf8') as f_out:
            first_line = True
            for line in f_in:

                # Comments + header
                if line.startswith('#'):
                    f_out.write(line)
                    continue
                if first_line:
                    f_out.write('# sentiment_lexicon=')
                    if sentiwordnet:
                        f_out.write('SentiWordNet')
                    else:
                        f_out.write('SentiWords')
                    f_out.write('\n')
                    first_line = False
                    labels = line.strip().split('\t')
                    try:
                        word_idx = labels.index('token')
                    except ValueError:
                        word_idx = 4
                    if sentiwordnet:
                        labels.append('positive')
                        labels.append('negative')
                    else:
                        labels.append('sentiment')
                    f_out.write('\t'.join(labels) + '\n')
                    continue

                line = line[:-1]  # Remove \n
                word = line.split('\t')[word_idx].lower()
                if sentiwordnet:
                    try:
                        value = lex[word]
                    except KeyError:
                        # Try looking up a lemmatized version
                        value = lex.get(nlp(word)[0].lemma_, (0.0, 0.0))
                    f_out.write(line + '\t' + str(value[0]) +
                                '\t' + str(value[1]) + '\n')
                else:
                    # SentiWords
                    try:
                        value = lex[word]
                    except KeyError:
                        # Try looking up a lemmatized version
                        value = lex.get(nlp(word)[0].lemma_, 0.0)
                    f_out.write(line + '\t' + str(value) + '\n')


def annotate_sequences(lex, in_file, nlp):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\thighest_pos\thighest_neg\n')
        for line in lines[1:]:
            text = line.split('\t')[4]
            highest_pos, highest_neg = 0.0, 0.0
            for word in text.strip().split():
                word_clean = ''
                if word.isalpha():
                    word_clean = word.lower()
                else:
                    for c in word.lower():
                        if c.isalpha():
                            word_clean += c
                if not word_clean:
                    continue
                try:
                    pos, neg = lex[word_clean]
                except KeyError:
                    # Try looking up a lemmatized version
                    pos, neg = lex.get(nlp(word_clean)[0].lemma_, (0.0, 0.0))
                if pos > highest_pos:
                    highest_pos = pos
                if neg > highest_neg:
                    highest_neg = neg
            f.write(line.strip() + '\t' + str(highest_pos) + '\t' +
                    str(highest_neg) + '\n')


if __name__ == '__main__':
    ### Task 1: Span identification
    lex = parse_sentiwordnet(SENTIWORDNET)
    nlp = English()
    annotate_tokens(lex, '../data/train-improved.tsv',
                    '../data/train-improved-sentiwordnet.tsv', nlp, True)
    annotate_tokens(lex, '../data/dev-improved.tsv',
                    '../data/dev-improved-sentiwordnet.tsv', nlp, True)
    annotate_tokens(lex, '../data/test-improved.tsv',
                    '../data/test-improved-sentiwordnet.tsv', nlp, True)

    ### Task 2: Technique identification
    lex = parse_sentiwordnet(SENTIWORDNET)
    nlp = English()
    annotate_sequences(lex, '../data/tc-train.tsv', nlp)
    annotate_sequences(lex, '../data/tc-dev.tsv', nlp)
    annotate_sequences(lex, '../data/tc-test.tsv', nlp)
