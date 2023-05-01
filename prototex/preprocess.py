import pathlib
from pathlib import Path
import numpy as np
import torch
from transformers import BartTokenizer
from args import args, datasets_config

def create_labels(dataset):
        temp=[ set(i)-set("O") for d in dataset[1] for i in d]
        return [ next(iter(i)) if len(i)>0 else "O"  for i in temp]


class CustomNonBinaryClassDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer: BartTokenizer,it_is_train=1,pos_or_neg=None,fix_seq_len=128,balance=False,
                 specific_label=None,for_protos=False):
        inputs = tokenizer(sentences, truncation=True, padding = 'max_length', max_length=fix_seq_len)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        labels = list(map(lambda x: datasets_config[args.data_dir]["classes"][x], labels))
        
        self.x = input_ids
        self.attn_mask = attention_mask
        self.y = labels
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.attn_mask[idx],self.y[idx]
    def collate_fn(self,batch):        
        return (torch.LongTensor([i[0] for i in batch]),
                torch.Tensor([i[1] for i in batch]),
                torch.LongTensor([i[2] for i in batch]))


## Preprocess function from the Propaganda Detection paper 
## https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html
def read_data(directory):
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text())
        labels.append(parse_label(f.as_posix().replace('.txt', '.labels.tsv')))
    # labels can be empty 
    return ids, texts, labels

def parse_label(label_path):
    labels = []
    f= Path(label_path)
    
    if not f.exists():
        return labels

    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append([int(parts[2]), int(parts[3]), parts[1], 0, 0])
    labels = sorted(labels) 

    if labels:
        length = max([label[1] for label in labels]) 
        visit = np.zeros(length)
        res = []
        for label in labels:
            if sum(visit[label[0]:label[1]]):
                label[3] = 1
            else:
               visit[label[0]:label[1]] = 1
            res.append(label)
        return res 
    else:
        return labels

def clean_text(articles, ids):
    texts = []
    for article, id in zip(articles, ids):
        sentences = article.split('\n')
        start = 0
        end = -1
        res = []
        for sentence in sentences:
           start = end + 1
           end = start + len(sentence)  # length of sequence 
           if sentence != "": # if not empty line
               res.append([id, sentence, start, end])
        texts.append(res)
    return texts

def make_dataset(directory):
    ids, texts, labels = read_data(directory)
    texts = clean_text(texts, ids)
    res = []
    for text, label in zip(texts, labels):
        # making positive examples
        tmp = [] 
        pos_ind = [0] * len(text)
        for l in label:
            for i, sen in enumerate(text):
                if l[0] >= sen[2] and l[0] < sen[3] and l[1] > sen[3]:
                    l[4] = 1
                    tmp.append(sen + [l[0], sen[3], l[2], l[3], l[4]])
                    pos_ind[i] = 1
                    l[0] = sen[3] + 1
                elif l[0] != l[1] and l[0] >= sen[2] and l[0] < sen[3] and l[1] <= sen[3]: 
                    tmp.append(sen + l)
                    pos_ind[i] = 1
        # making negative examples
        dummy = [0, 0, 'O', 0, 0]
        for k, sen in enumerate(text):
            if pos_ind[k] != 1:
                tmp.append(sen+dummy)
        res.append(tmp)     
    return res
        
def make_bert_testset(dataset):

    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        tmp_sen = article[0][1]
        tmp_i = article[0][0]
        label = ['O'] * len(tmp_sen.split(' '))
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                if tmp_sen != sentence[1]:
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                    label = ['O'] * len(token_len)
                start = sentence[4] - sentence[2] 
                end = sentence[5] - sentence[2]
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)): 
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end) 
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else: 
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_sen = sentence[1]
                tmp_i = sentence[0]
            else:
                tmp_doc.append(tokens)
                tmp_id.append(sentence[0])
        if len(sentence) == 9:
            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
        words.append(tmp_doc) 
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids


def make_bert_dataset(dataset):
    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        tmp_sen = article[0][1]
        tmp_i = article[0][0]
        label = ['O'] * len(tmp_sen.split(' '))
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                if tmp_sen != sentence[1] or sentence[7]:
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                    if tmp_sen != sentence[1]:
                        label = ['O'] * len(token_len)
                start = sentence[4] - sentence[2] 
                end = sentence[5] - sentence[2]
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)): 
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)  
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else: 
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_sen = sentence[1]
                tmp_i = sentence[0]
            else:
                tmp_doc.append(tokens)
                tmp_id.append(sentence[0])
        if len(sentence) == 9:
            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
        words.append(tmp_doc) 
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids


def mda(dataset):
    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                start = sentence[4] - sentence[2]
                end = sentence[5] - sentence[2]
                label = ['O'] * len(token_len)
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)):
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)  
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else:
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_label.append(label)
            tmp_doc.append(tokens)
            tmp_id.append(sentence[0])
        words.append(tmp_doc)
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids

