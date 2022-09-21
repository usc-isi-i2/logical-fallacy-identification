"""
Repetition features for the technique classification task.

Encodes if (and how often) a given text fragment is repeated in a news article,
and if so whether the given span is the first occurrence in the article or a
later repetition.
"""
import os
import string
import spacy

TC_LABELS_FILE = "../datasets/train-task2-TC.labels"
TC_LABELS_FILE_DEV = "../datasets/dev-task-TC-template.out"
TC_LABELS_FILE_TEST = "../datasets/test-task-TC-template.out"
TRAIN_DATA_FOLDER = "../datasets/train-articles/"
DEV_DATA_FOLDER = "../datasets/dev-articles/"
TEST_DATA_FOLDER = "../datasets/test-articles/"


def get_repetition_features(labels_file, raw_data_folder, file_to_write,
                            n_of_repetitions=True,
                            not_first_occurrence=True,
                            n_of_lemmatized_repetitions=True,
                            verbose=True):
    """
    Function which creates the repetitions features for the data.
    The function:
    - check the number of repetitions (count - 1) of the span in the article;
    - check that it is not the first occurrence of the span in the article;
    - check the number of the repetitions of the lemmatized span in the
      lemmatized article;
    - apply advanced repetition control by checking whether some words from the
      span occur in the n-grams of the similar length in the article.
    """
    with open(labels_file, encoding='utf8') as f:
        table = f.readlines()
        table = [row.split() for row in table]

    sp = spacy.load('en_core_web_sm')

    open_doc_id = ""
    open_doc_txt = ""
    output_table = []
    instances_count = 0

    header = ['document_id', 'label', 'span_start', 'span_end', 'text',
              'n_of_repetitions', 'not_first_occurrence',
              'n_of_lemmatized_repetitions']
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
                norm_article = normalize_string(open_doc_txt)
                open_doc_id = doc_id

                if n_of_lemmatized_repetitions:
                    spacy_text = sp(norm_article)
                    lem_text = " ".join([word.lemma_ if word.lemma_ != "-PRON-"
                                         else word.text
                                         for word in spacy_text])

        raw_span = open_doc_txt[from_id:to_id].strip().replace("\n", " ")
        span = normalize_string(raw_span)

        output_features = [doc_id, label, str(from_id), str(to_id), raw_span]

        if n_of_repetitions:
            output_features.append(str(norm_article.count(span) - 1))

        if not_first_occurrence:
            # Checking whether it is not the first occurrence of the given span
            # -> we use a small window around the index of the span beginning
            value = open_doc_txt.find(raw_span)\
                not in range(from_id - 3, from_id + 3)
            output_features.append(str(int(value)))

        if n_of_lemmatized_repetitions:
            spacy_span = sp(span)
            lem_span = " ".join([word.lemma_ if word.lemma_ != "-PRON-"
                                 else word.text for word in spacy_span])
            output_features.append(str(max(lem_text.count(lem_span) - 1, 0)))

        if label != "Repetition":
            output_table.append(output_features)
        instances_count += 1

        if verbose and (instances_count % 100 == 0):
            print("Processed {} instances".format(instances_count))

    with open(file_to_write, 'w', encoding='utf8') as f:
        for row in output_table:
            f.write('\t'.join(row) + "\n")


def normalize_string(input):
    """
    Helper function to:
    1. remove the punctuation;
    2. remove some quotes which are left;
    3. replace any new line characters;
    4. strip;
    5. transform to lower case.

    :param input: input string of the entire article or a span
    :return: a normalized string
    """
    return input.translate(str.maketrans('', '', string.punctuation))\
        .translate(str.maketrans('', '', '“”‘’')).replace("\n", " ")\
        .strip().lower()


if __name__ == '__main__':
    get_repetition_features(TC_LABELS_FILE, TRAIN_DATA_FOLDER,
                            "../data/tc-train-complex-repetitions-short.tsv")
    get_repetition_features(TC_LABELS_FILE_DEV, DEV_DATA_FOLDER,
                            "../data/tc-dev-complex-repetitions.tsv")
    get_repetition_features(TC_LABELS_FILE_TEST, TEST_DATA_FOLDER,
                            "../data/tc-test-complex-repetitions.tsv")
