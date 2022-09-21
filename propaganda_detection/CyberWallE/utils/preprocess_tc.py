"""
Preprocessing the datasets for task 2: technique classification.
"""
import os


TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TC_LABELS_FILE_DEV = "../datasets/dev-task-TC-template.out"
TC_LABELS_FILE_TEST = "../datasets/test-task-TC-template.out"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
DEV_DATA_FOLDER = "../datasets/dev-articles/"
TEST_DATA_FOLDER = "../datasets/test-articles/"


def get_spans_from_text(labels_file, raw_data_folder, file_to_write,
                        add_repetition_count=False, add_repetition_text=False,
                        context=None):
    """
    Subtracts spans from raw texts and creates a new file
    which contains both labels and spans.

    :param labels_file: dir of the tab-separated file of form
        document_id    propaganda_label    beginning of span    end of span
    :param raw_data_folder: dir of folder with texts
    :param file_to_write: directory of the file to write
    :param add_repetition_count: add the number of the span occurrences in the given document as a feature
    :param add_repetition_text:
    :param context: add the context in which occurred the span to the text of the span separated by [SEP]
    """
    with open(labels_file, encoding='utf8') as f:
        table = f.readlines()
        table = [row.split() for row in table]

    open_doc_id = ""
    open_doc_txt = ""
    output_table = []

    header = ['document_id', 'label', 'span_start', 'span_end', 'text']
    if add_repetition_count:
        header.append('repetitions')
    output_table.append(header)

    for row in table:
        doc_id = row[0]
        label = row[1]
        from_id = int(row[2])        # idx of the beginning of the span
        to_id = int(row[3])          # idx of the end of the span

        # read the file if it's not opened yet
        if str(doc_id) != open_doc_id:
            with open(os.path.join(raw_data_folder,
                                   "article{}.txt".format(doc_id)),
                      encoding='utf8') as f:
                open_doc_txt = f.read()
                open_doc_id = doc_id

        span = open_doc_txt[from_id:to_id].strip()
        text = span.replace('\n', ' ')

        if context is None:
            if add_repetition_count or add_repetition_text:
                n_reps = open_doc_txt.count(span)
                if add_repetition_text and n_reps > 1:
                    text += ' ' + text
            if add_repetition_count:
                # -1 to count repetitions instead of occurrences
                output_table.append(row + [text] + [str(n_reps - 1)])
            else:
                output_table.append(row + [text])

        if context == "sentence":
            if label == "Repetition":
                text = text + " [SEP] " + text
            else:
                text = text + " [SEP] " + get_context(open_doc_txt, text,
                                                      from_id, to_id)
            output_table.append(row + [text])

    with open(file_to_write, 'w', encoding='utf8') as f:
        for row in output_table:
            f.write('\t'.join(row) + "\n")


def get_context(open_doc_txt, span_txt, from_id, to_id):
    output_prefix = ""
    output_suffix = ""

    for c in reversed(open_doc_txt[:from_id]):
        if c in "!?." or c == "\n":
            output_prefix = "".join(reversed(output_prefix))
            break
        else:
            output_prefix += c

    for c in open_doc_txt[to_id:]:
        output_suffix += c
        if c in "!?." or c == "\n":
            break
            output_prefix += output_prefix

    return (output_prefix + span_txt + output_suffix).replace("\n", "")


def add_repetition_to_text(file_to_read, file_to_write):
    with open(file_to_read, "r", encoding='utf8') as fl:
        lines = fl.readlines()

    with open(file_to_write, "w", encoding='utf8') as fl:
        for line in lines:
            columns = line.strip().split("\t")
            if columns[1] == "Repetition":
                columns[4] = columns[4] + " " + columns[4]
            fl.write("\t".join(columns) + "\n")


def add_sequence_lengths(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tlength\n')
        for line in lines[1:]:
            text = line.split('\t')[4]
            length = len(text.strip().split())
            f.write(line.strip() + '\t' + str(length) + '\n')


def add_question_marks(in_file):
    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\tquestion\n')
        for line in lines[1:]:
            f.write(line.strip() + '\t')
            text = line.split('\t')[4]
            if '?' in text:
                f.write('1\n')
            else:
                f.write('0\n')


if __name__ == '__main__':
    ### For the 100% BERT model:
    add_repetition_to_text("../data/tc-train.tsv",
                           "../data/tc-train-repetition.tsv")
    get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER,
                        "../data/tc-train-context.tsv",
                        context="sentence")
    get_spans_from_text(TC_LABELS_FILE_DEV, DEV_DATA_FOLDER,
                        "../data/tc-dev-repetition.tsv",
                        add_repetition_text=True)
    get_spans_from_text(TC_LABELS_FILE_TEST, TEST_DATA_FOLDER,
                        "../data/tc-test-repetition.tsv",
                        add_repetition_text=True)

    ### For the composite models:
    get_spans_from_text(TC_LABELS_FILE, TRAIN_DATA_FOLDER,
                        "../data/tc-train.tsv", add_repetition_count=True)
    get_spans_from_text(TC_LABELS_FILE_DEV, DEV_DATA_FOLDER,
                        "../data/tc-dev.tsv", add_repetition_count=True)
    get_spans_from_text(TC_LABELS_FILE_TEST, TEST_DATA_FOLDER,
                        "../data/tc-test.tsv", add_repetition_count=True)

    add_sequence_lengths('../data/tc-train.tsv')
    add_sequence_lengths('../data/tc-dev.tsv')
    add_sequence_lengths('../data/tc-test.tsv')

    add_question_marks('../data/tc-train.tsv')
    add_question_marks('../data/tc-dev.tsv')
    add_question_marks('../data/tc-test.tsv')
