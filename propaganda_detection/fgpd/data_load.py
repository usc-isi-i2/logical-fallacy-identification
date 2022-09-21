import numpy as np
import torch
from torch.utils import data
import pathlib
from preprocess import make_dataset, make_bert_dataset, make_bert_testset
from pytorch_pretrained_bert import BertTokenizer
from hp import hp

if hp.bert:
    num_task = 1
    masking = 0 
    hier = 0
elif hp.joint:
    num_task = 2
    masking = 0 
    hier = 0
elif hp.granu:
    num_task = 2
    masking = 0 
    hier = 1 
elif hp.mgn:
    num_task = 2
    masking = 1
    hier = 0

if hp.sig:
    sig = 1
    rel = 0 
elif hp.rel:
    sig = 0 
    rel = 1
    
input_size=768
VOCAB, tag2idx, idx2tag = [], [], []

if num_task == 1:

    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt"
                  , "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language"
                  , "Reductio_ad_hitlerum", "Bandwagon"
                  , "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy"
                  , "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"))

#sentence classification
if num_task == 2:
    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt"
                  , "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language"
                  , "Reductio_ad_hitlerum", "Bandwagon"
                  , "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy"
                  , "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"))
    VOCAB.append(("Non-prop", "Prop"))

for i in range(num_task):
    tag2idx.append({tag:idx for idx, tag in enumerate(VOCAB[i])})
    idx2tag.append({idx:tag for idx, tag in enumerate(VOCAB[i])})

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

class PropDataset(data.Dataset):
    def __init__(self, fpath, IsTest=False):

        directory = pathlib.Path(fpath)
        dataset = make_dataset(directory)
        if IsTest:
            words, tags, ids = make_bert_testset(dataset)
        else:
            words, tags, ids = make_bert_dataset(dataset)
        flat_words, flat_tags, flat_ids = [], [], []
        for article_w, article_t, article_id in zip(words, tags, ids):
            for sentence, tag, id in zip(article_w, article_t, article_id):
                flat_words.append(sentence)
                flat_tags.append(tag)
                flat_ids.append(id)

        sents, ids = [], [] 
        tags_li = [[] for _ in range(num_task)]
   
        for word, tag, id in zip(flat_words, flat_tags, flat_ids):
            words = word
            tags = tag

            ids.append([id])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tmp_tags = []

            if num_task != 2:
                for i in range(num_task):
                    tmp_tags.append(['O']*len(tags))
                    for j, tag in enumerate(tags):
                        if tag != 'O' and tag in VOCAB[i]:
                            tmp_tags[i][j] = tag
                    tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])
            elif num_task == 2:
                tmp_tags.append(['O']*len(tags))
                tmp_tags.append(['Non-prop'])
                for j, tag in enumerate(tags):
                    if tag != 'O' and tag in VOCAB[0]:
                        tmp_tags[0][j] = tag
                        tmp_tags[1] = ['Prop']
                for i in range(num_task):
                    tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])

        self.sents, self.ids, self.tags_li = sents, ids, tags_li
        assert len(sents) == len(ids) == len(tags_li[0])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = self.sents[idx] # tokens, tags: string list
        ids = self.ids[idx] # tokens, tags: string list
        tags = list(list(zip(*self.tags_li))[idx])

        x, is_heads = [], [] # list of ids
        y = [[] for _ in range(num_task)] # list of lists of lists
        tt = [[] for _ in range(num_task)] # list of lists of lists
        if num_task != 2:
            for w, *t in zip(words, *tags):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
    
                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))
    
                t = [[t[i]] + [t[i]] * (len(tokens) - 1) for i in range(num_task)]

                y_tmp = []
                for i in range(num_task):
                    y[i].extend([tag2idx[i][each] for each in t[i]])
                    tt[i].extend(t[i])

                x.extend(xx)
                is_heads.extend(is_head)
    
        elif masking or num_task == 2:
            for w, t in zip(words, tags[0]):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)

                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))

                t = [t] + [t] * (len(tokens) - 1)
                y[0].extend([tag2idx[0][each] for each in t])
                tt[0].extend(t)

                x.extend(xx)
                is_heads.extend(is_head)
            if tags[1][1] == 'Non-prop':
                y[1].extend([1, 0])
                tt[1].extend(['Non-prop'])
            elif tags[1][1] == 'Prop':
                y[1].extend([0, 1])
                tt[1].extend(['Prop'])

        seqlen = len(y[0])

        words = " ".join(ids + words)

        for i in range(num_task):
            tags[i]= " ".join(tags[i]) 

        att_mask = [1] * seqlen
        return words, x, is_heads, att_mask, tags, y, seqlen

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))

    att_mask = f(-4, maxlen)
    y = []
    tags = []

    if num_task !=2:
        for i in range(num_task):
            y.append(torch.LongTensor([sample[-2][i] + [0] * (maxlen-len(sample[-2][i])) for sample in batch]))
            tags.append([sample[-3][i] for sample in batch])
    else:
        y.append(torch.LongTensor([sample[-2][0] + [0] * (maxlen-len(sample[-2][0])) for sample in batch]))
        y.append(torch.LongTensor([sample[-2][1] for sample in batch]))
        for i in range(num_task):
            tags.append([sample[-3][i] for sample in batch])


    return words, x, is_heads, att_mask, tags, y, seqlen

