"""
Using BertForTokenClassification for preliminary experiments for the
Span Identification task. We did not pursue this approach further.

---

Much of the code is based on this tutorial by Tobias Sterbak:
https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
(MIT license)

---

License for this file: https://choosealicense.com/licenses/mit/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# !pip install pytorch_pretrained_bert
# !pip install seqeval

import torch
import urllib
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from keras.preprocessing.sequence import pad_sequences
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score


### CONFIG ###
MAX_LEN = 50
BATCH_SIZE = 32
TRAIN_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/train-improved-sentiwordnet-arguingfullindiv-pos.tsv?token=AD7GEDK3MI27HVJPQWOE74C6FBZHA'
DEV_URL = 'https://raw.githubusercontent.com/cicl-iscl/CyberWallE/master/data/dev-improved-sentiwordnet-arguingfullindiv-pos.tsv?token=AD7GEDM3LOMZM6MP4HZS4MS6FBZHK'
EPOCHS = 3
MAX_GRAD_NORM = 1.0
##############


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                   num_labels=2)
model.cuda()


def get_comments(filename, url=True):
    if url:
        comments = []
        with urllib.request.urlopen(filename) as f:
            for line in f:
                if line.startswith(b'#'):
                    comments.append(line.decode("utf-8"))
                else:
                    break
        return comments
    with open(filename, 'r', encoding='utf8') as f:
        commentiter = takewhile(lambda s: s.startswith('#'), f)
        comments = list(commentiter)
    return comments


def get_data(url, get_labels=True)
    comments = get_comments(url)
    df = pd.read_csv(url, sep='\t', skiprows=len(comments), quoting=3)
    input_df = df.groupby('sent_id')['token'].apply(list).to_frame()
    sentences = input_df['token'].tolist()
    sentences = [['[CLS]'] + [str(word).lower() for word in sent] + ['[SEP]'] for sent in sentences]
    print(sentences[0])
    tokenized_texts = [tokenizer.tokenize(' '.join(sent)) for sent in sentences]
    print(tokenized_texts[0])
    print(tokenized_texts[8])
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post",
                              padding="post")
    labels = None
    if get_labels:
        tag2idx = {'O': 0, 'B': 1, 'I': 1}
        labels = df.groupby('sent_id')['label'].apply(list).to_frame()['label'].tolist()
        print(labels[0])
        labels = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                                maxlen=MAX_LEN, value=tag2idx["O"],
                               padding="post", dtype="long", truncating="post")
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    return input_ids, attention_masks, labels


input_ids, attention_masks, labels = get_data(TRAIN_URL)
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018,
                                                            test_size=0.1)
train_data = TensorDataset(torch.tensor(tr_inputs), torch.tensor(tr_masks),
                           torch.tensor(tr_tags))
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data),
                              batch_size=BATCH_SIZE)

tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
valid_data = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_tags),
                           torch.tensor(val_masks))
valid_dataloader = DataLoader(valid_data,
                              sampler=SequentialSampler(valid_data),
                              batch_size=BATCH_SIZE)

label_order = ['O', 'I']

for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
    for step, batch in enumerate(train_dataloader):
        # Use GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Forward + backward passes
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                       max_norm=MAX_GRAD_NORM)
        # Update parameters
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():  # Save memory
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    print("Validation loss: {}".format(eval_loss / nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_tags = [label_order[p_i] for p in predictions for p_i in p]
    valid_tags = [label_order[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print(pred_tags)
    print(valid_tags)

input_ids, attention_masks, _ = get_data(DEV_URL, False)
pred_data = TensorDataset(torch.tensor(input_ids),
                          torch.tensor(attention_masks))
pred_dataloader = DataLoader(pred_data, sampler=SequentialSampler(pred_data),
                             batch_size=BATCH_SIZE)

model.eval()
predictions = []

for batch in pred_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
preds = []
for mask, pred in zip(attention_masks, predictions):
    sent_pred = []
    for m, p in zip(mask, pred):
        if m > 0.0:
            sent_pred.append(label_order[p])
    preds.append(sent_pred)
dev_input['n_toks'] = dev_input['token'].apply(lambda x: len(x))
dev_input.head()

preds_with_toolong = []
for row in dev_input.itertuples():
    sent_idx = row.Index - 1
    sent_pred = preds[sent_idx]
    n_actual_toks = row.n_toks
    n_pred_toks = len(sent_pred)

    pred_without_subwordtoks = []
    for tok, pred in zip(tokenized_texts[sent_idx], sent_pred):
        if tok == '[CLS]' or tok == '[SEP]':
            continue
        if tok.startswith('##'):
            pred_without_subwordtoks[-1] = pred
        else:
            pred_without_subwordtoks.append(pred)

    for _ in range(n_actual_toks - len(pred_without_subwordtoks)):
        pred_without_subwordtoks.append('O')
    preds_with_toolong.append(pred_without_subwordtoks)

print(len(preds_with_toolong))
flat_predictions = [item for sublist in preds_with_toolong for item in sublist]
print(len(flat_predictions))


def si_predictions_to_spans(label_df):
    spans = []
    prev_label = 'O'
    prev_span_start = '-1'
    prev_span_end = '-1'
    prev_article = ''

    for row in label_df.itertuples():
        article = row.document_id
        span_start = row.token_start
        span_end = row.token_end
        label = row.label

        span, prev_span_start = update_prediction(article, label,
                                                  span_start, span_end,
                                                  prev_article, prev_label,
                                                  prev_span_start,
                                                  prev_span_end)
        if span is not None:
            spans.append(span)

        prev_article = article
        prev_label = label
        prev_span_end = span_end

    # Make sure we get the last prediction
    span, _ = update_prediction(article, label, span_start, span_end,
                                prev_article, prev_label, prev_span_start,
                                prev_span_end)
    if span is not None:
        spans.append(span)
    return spans


# Helper method for si_predictions_to_spans
def update_prediction(article, label, span_start, span_end, prev_article,
                      prev_label, prev_span_start, prev_span_end):
    span = None
    cur_span_start = prev_span_start
    # Ending a span: I-O, B-O, I-B, B-B, new article
    if prev_label != 'O' and (label != 'I' or prev_article != article):
        span = (prev_article, prev_span_start, prev_span_end)

    # Starting a new span: O-B, O-I, I-B, B-B, new article
    if label == 'B' or (label == 'I' and prev_label == 'O') \
            or prev_article != article:
        # Update the start of the current label span
        cur_span_start = span_start
    return span, cur_span_start


dev_df["label"] = flat_predictions
print(dev_df.head())
dev_df.to_csv("dev-task-SI.out", sep='\t', header=True, index=False)
spans = si_predictions_to_spans(dev_df)
with open('bert-for-token-classification.txt', mode='w') as f:
    for span in spans:
        f.write(str(span[0]) + '\t' + str(span[1]) + '\t' +
                str(span[2]) + '\n')
