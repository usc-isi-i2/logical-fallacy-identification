"""
Preprocessing the datasets for task 1: span identification.
"""
import os
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TC_LABELS_FILE_DEV = "../datasets/dev-task-TC-template.out"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
DEV_DATA_FOLDER = "../datasets/dev-articles/"
TEST_DATA_FOLDER = "../datasets/test-articles/"
SI_PREDICTIONS_FILE = '../data/dev_predictions_bio.tsv'
SI_SPANS_FILE = '../data/dev_predictions_spans.txt'
LABELS_DATA_FOLDER = "../datasets/train-labels-task2-technique-classification/"
GENERATED_LABELS_FOLDER = "../datasets/dev-labels-task2-technique-classification/"


def annotate_text(raw_data_folder, labels_data_folder, file_to_write,
                  max_sent_len=35, improved_sent_splitting=True,
                  training=True):
    """
    Creates a token-level input file for the span identification task and adds
    sentence IDs to the tokens.
    """
    # max_sent_len = -1 ==> no sentence splitting
    if max_sent_len == -1:
        # the corresponding if-block can handle this
        improved_sent_splitting = True
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    if improved_sent_splitting:
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc',
                                        'ms', 'rep', 'u.s', 'feb', 'sen'])
        splitter = PunktSentenceTokenizer(punkt_param)
        splitter.PUNCTUATION = tuple(';:,.!?"')
    output_table = []
    file_counter = 0
    sent_no_total = 0

    print("Total number of files - {}".format(
        len(os.listdir(raw_data_folder))))

    # Reading all the files from the raw text directory
    article_file_names = [file_name for file_name in
                          os.listdir(raw_data_folder)
                          if file_name.endswith(".txt")]
    article_file_names.sort()

    for file_name in article_file_names:
        if training:
            label_file_name = file_name.replace(".txt", ".task2-TC.labels")
            print("raw_article: {}\tlabel_file: {}".format(file_name,
                                                           label_file_name))

            # Read the labels file with 4 columns of format
            # doc_id : label_of_span : idx_span_begin : idx_span_end
            with open(os.path.join(labels_data_folder, label_file_name),
                      encoding="utf-8") as file:
                rows = file.readlines()
                rows = [row.strip().split("\t") for row in rows
                        if len(row.split("\t")) == 4]

                # Saving mappings char_idx->labels into the dictionary
                char_idx2label = dict()
                for row in rows:
                    label = row[1]
                    idx_from = int(row[2])
                    idx_to = int(row[3])

                    for idx in range(idx_from, idx_to):
                        if idx not in char_idx2label.keys():
                            char_idx2label[idx] = []
                        char_idx2label[idx].append(label)
        else:
            print("raw_article: " + file_name)

        # Read the article and process the text
        with open(os.path.join(raw_data_folder, file_name),
                  encoding="utf-8") as file:
            file_text = file.readlines()
            # Keep linebreaks for better sentence splitting
            file_text = ''.join([line for line in file_text])

            # Normalizing punctuation marks to help the tokenizer.
            file_text = file_text.replace('“', '"').replace('”', '"')
            file_text = file_text.replace("’", "'").replace("‘", "'")

            sentences = []
            if improved_sent_splitting:
                # Line breaks -> helps with headlines
                paragraphs = file_text.split('\n')
                for para in paragraphs:
                    para = para.strip()
                    sentences_raw = splitter.sentences_from_text(para)
                    for sent in sentences_raw:
                        sent = sent.strip()
                        tokens = tokenizer(sent)
                        if len(tokens) <= max_sent_len or max_sent_len == -1:
                            # No need to split the sentence!
                            if len(sent) == 0:
                                # Can happen when paragraphs are separated by
                                # several line breaks.
                                continue
                            sentences.append(sent)
                            continue

                        # Try splitting based on quotes.
                        quote_fragments, all_ok = punct_based_split_sent(
                            tokenizer, sent, max_sent_len, '"')
                        if all_ok:
                            sentences += quote_fragments
                            continue

                        # Other punctuation for splitting: ; :
                        for quote_frag in quote_fragments:
                            semicolon_fragments, all_ok =\
                                punct_based_split_sent(tokenizer, quote_frag,
                                                       max_sent_len, ';')
                            if all_ok:
                                sentences += semicolon_fragments
                                continue

                            for semicolon_frag in semicolon_fragments:
                                colon_fragments, all_ok =\
                                    punct_based_split_sent(tokenizer,
                                                           semicolon_frag,
                                                           max_sent_len, ':')
                                if all_ok:
                                    sentences += colon_fragments
                                    continue

                                # Commas:
                                for col_frag in colon_fragments:
                                    comma_fragments, all_ok =\
                                        punct_based_split_sent(tokenizer,
                                                               col_frag,
                                                               max_sent_len,
                                                               ',')
                                    if all_ok:
                                        sentences += comma_fragments
                                        continue

                                    # Last resort:
                                    # Split after max_sent_len tokens
                                    for comma_frag in comma_fragments:
                                        sentences += forcefully_split_sent(
                                            tokenizer, comma_frag,
                                            max_sent_len)
            else:
                # Cut long sentences into fragments that are (up to)
                # max_sent_len characters long
                # (the last fragment in a sentence might be shorter)
                file_text = file_text.replace('\n', ' ')
                sentences_raw = sent_tokenize(file_text)
                for sent in sentences_raw:
                    sentences += forcefully_split_sent(tokenizer, sent,
                                                       max_sent_len)

            i = 0
            for sent in sentences:
                sent = sent.strip()
                i = file_text.find(sent, i)
                max_idx = i + len(sent)

                if sent == '':
                    continue

                if improved_sent_splitting:
                    if len(sent.strip()) < 2:  # single char noise
                        continue

                sent_no_total += 1
                for token in tokenizer(sent):
                    token = str(token)
                    token_idx = file_text.find(token, i, max_idx)
                    i = token_idx + len(token)
                    output = [file_name.replace("article", "")
                                       .replace(".txt", ""),
                              str(sent_no_total),
                              str(token_idx),
                              str(i),
                              token]
                    if training:
                        # Check the label of the corresponding char_idx
                        label = char_idx2label.get(token_idx, ['None'])
                        output.append("|".join(label))
                    output_table.append(output)

        file_counter += 1
        print("Finished {} files\n".format(file_counter))

        with open(file_to_write, 'w', encoding="utf-8") as f:
            f.write('# max_sent_len=' + str(max_sent_len) +
                    ', improved_sent_splitting=' +
                    str(improved_sent_splitting) + '\n')
            f.write('document_id\tsent_id\ttoken_start\ttoken_end\ttoken')
            if training:
                f.write('\tlabel')
            f.write('\n')
            for row in output_table:
                f.write('\t'.join(row) + "\n")


# Helper method for annotate_text
def forcefully_split_sent(tokenizer, sent, max_sent_len):
    sentences = []
    tokens = tokenizer(sent)
    n_toks = len(tokens)
    if n_toks <= max_sent_len:
        if n_toks == 0:
            return []
        sentences.append(sent.strip())
        return sentences

    tok_idx = 0
    fragment_start = 0
    fragment_end = 0
    while tok_idx < n_toks:
        # This is so hacky >:(
        for token in tokens[tok_idx:tok_idx + max_sent_len]:
            fragment_end = sent.find(str(token),
                                     fragment_end) + len(token)
        sentences.append(sent[fragment_start:fragment_end]
                         .strip())
        tok_idx += max_sent_len
        fragment_start = fragment_end
    return sentences


# Helper method for annotate_text
def punct_based_split_sent(tokenizer, sent, max_sent_len, punct):
    if punct not in sent:
        return [sent], False
    sents = []
    tokens = tokenizer(sent)
    if len(tokens) <= max_sent_len:
        sents.append(sent)
        return sents, True
    # Try splitting along the punctuation mark.
    sent_fragments = sent.split(punct)
    n_frags = len(sent_fragments)
    prev_len = max_sent_len
    longest = 0
    for i, sent_fragment in enumerate(sent_fragments):
        if n_frags > 1 and i < n_frags - 1:
            sent_fragment += punct

        if len(sent_fragment.strip()) == 0:
            continue

        cur_len = len(tokenizer(sent_fragment.strip()))
        if cur_len > longest:
            longest = cur_len
        # We don't want to end up with a ton of very short sentences.
        if cur_len + prev_len <= max_sent_len:
            sents[-1] = sents[-1] + sent_fragment
            prev_len += cur_len
        else:
            sents.append(sent_fragment)
            prev_len = cur_len
    return sents, longest <= max_sent_len


# Helper method for labels2bio
def overlap(l1, l2):
    if l1 == l2:
        return True
    if l1 in l2:
        return True
    if l2 in l1:
        return True
    return False


def labels2bio(span_file, bio_file):
    """
    Changes labels from category-specific labels to BIO-style labels.
    """
    with open(span_file, encoding='utf8') as infile:
        with open(bio_file, 'w', encoding='utf8') as outfile:
            prev_label = 'None'
            prev_article = '-1'
            first_line = True
            for line in infile:

                # Comments + header
                if line.startswith('#'):
                    outfile.write(line)
                    continue
                if first_line:
                    first_line = False
                    outfile.write(line)
                    labels = line.strip().split('\t')
                    try:
                        doc_idx = labels.index('document_id')
                    except ValueError:
                        doc_idx = 0
                    try:
                        label_idx = labels.index('label')
                    except ValueError:
                        label_idx = len(labels) - 1
                    continue

                fields = line.strip().split('\t')
                article = fields[doc_idx]
                label = fields[label_idx]

                if label == 'None':
                    bio_label = 'O'
                elif overlap(prev_label, label) and prev_article == article:
                    bio_label = 'I'
                else:
                    bio_label = 'B'
                prev_label = label
                prev_article = article

                fields[label_idx] = bio_label
                outfile.write('\t'.join(fields))
                outfile.write('\n')


def get_si_dev_gs(tc_file='../datasets/dev-task-TC-template.out',
                  outfile='../data/si-dev-GS.txt'):
    """
    Extracts the gold-standard dev set labels for the span identification task
    from the dev input for the technique classification task.
    """
    articles2spans = {}
    with open(tc_file, encoding='utf8') as f_in:
        for line in f_in:
            fields = line[:-1].split('\t')
            article = fields[0]
            span_start = int(fields[2])
            span_end = int(fields[3])
            try:
                spans = articles2spans[article]
            except KeyError:
                spans = []
            spans.append((span_start, span_end))
            articles2spans[article] = spans

    rows = []
    articles = sorted([art for art in articles2spans])
    for article in articles:
        spans = articles2spans[article]
        spans.sort(key=lambda tup: tup[0])
        rows_in_article = []
        for span in spans:
            span = (article, span[0], span[1])
            if not rows_in_article:
                rows_in_article = [span]
                continue
            prev_span = rows_in_article[-1]
            if span[1] <= prev_span[2]:
                if span[2] <= prev_span[2]:
                    continue
                rows_in_article[-1] = (article, prev_span[1], span[2])
            else:
                rows_in_article.append(span)
        rows += rows_in_article

    with open(outfile, 'w', encoding='utf8') as f_out:
        for row in rows:
            f_out.write(row[0] + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n')


def generate_labels_folder(file_with_labels, new_folder_dir):
    """
    Creates a folder which contains individual files with labels that can be
    used as labels_data_folder argument for the annotate_data function.

    :param file_with_labels: in our case, dev-task-TC-template.out
    :param new_folder_dir:
    """
    document_id2spans = dict()

    with open(file_with_labels, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            document_id, _, span_start, span_end = line.split()

            if document_id not in document_id2spans:
                document_id2spans[document_id] = []

            document_id2spans[document_id].append((span_start, span_end))

    for document_id in document_id2spans.keys():
        new_filename = "{}article{}.task2-TC.labels".format(new_folder_dir,
                                                            document_id)
        with open(new_filename, "w") as file:
            for span_start, span_end in document_id2spans[document_id]:
                output_line = "{}\tDumb_class\t{}\t{}\n".format(document_id,
                                                                span_start,
                                                                span_end)
                file.write(output_line)


if __name__ == '__main__':
    # Preprocessing the training set as input for fitting the model, and
    # the development/test sets as input for predictions:
    annotate_text(TRAIN_DATA_FOLDER, LABELS_DATA_FOLDER,
                  '../data/si-train-FULL-LABELS.tsv',
                  improved_sent_splitting=True)
    labels2bio('../data/si-train-FULL-LABELS.tsv',
               '../data/si-train.tsv')
    annotate_text(DEV_DATA_FOLDER, None,
                  "../data/si-dev.tsv",
                  improved_sent_splitting=True,
                  training=False)
    annotate_text(TEST_DATA_FOLDER, None,
                  "../data/si-test.tsv",
                  improved_sent_splitting=True,
                  training=False)

    # Dev set labels for checking model performance without having to upload
    # predictions:
    get_si_dev_gs()
    # For feature ablation studies:
    get_si_dev_gs(tc_file='../datasets/test-task-TC-template.out',
                  outfile='../data/si-test-GS.txt')

    # Dev set as input for training:
    generate_labels_folder(TC_LABELS_FILE_DEV, GENERATED_LABELS_FOLDER)
    annotate_text(DEV_DATA_FOLDER, GENERATED_LABELS_FOLDER,
                  "../data/si-train+dev-FULL-LABELS.tsv",
                  improved_sent_splitting=True)
    labels2bio('../data/si-train+dev-FULL-LABELS.tsv',
               '../data/si-train+dev.tsv')
