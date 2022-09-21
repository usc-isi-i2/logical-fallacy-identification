import pathlib
from pathlib import Path
import numpy as np
import torch
# from pytorch_pretrained_bert import BertTokenizer

labels_set={'Appeal_to_Authority',
 'Appeal_to_fear-prejudice',
 'Bandwagon',
 'Black-and-White_Fallacy',
 'Causal_Oversimplification',
 'Doubt',
 'Exaggeration,Minimisation',
 'Flag-Waving',
 'Loaded_Language',
 'Name_Calling,Labeling',
 'O',
 'Obfuscation,Intentional_Vagueness,Confusion',
 'Red_Herring',
 'Reductio_ad_hitlerum',
 'Repetition',
 'Slogans',
 'Straw_Men',
 'Thought-terminating_Cliches',
 'Whataboutism'}

## Additional classes and functions for data preparation for ProtoTexNL

def create_labels(dataset):
        temp=[ set(i)-set("O") for d in dataset[1] for i in d]
        return [ next(iter(i)) if len(i)>0 else "O"  for i in temp]

class BinaryClassDataset(torch.utils.data.Dataset):
    def __init__(self, x,y,y_txt,tokenizer,it_is_train=1,pos_or_neg=None,fix_seq_len=256,balance=False,
                 specific_label=None,for_protos=False):
#         temp=tokenizer(x)
#         self.input_ids,self.attention_mask=temp["input_ids"],temp["attention_mask"]
        self.x=[]
        self.attn_mask=[]
        self.labels_mask=[]
        self.y_txt=[]
        self.y=[]
        self.labels_ids={}
        for i in labels_set:
            self.labels_ids[i]=len(self.labels_ids)
        self.y_fine_int=[]
        it_is_train_proxy=it_is_train
        for split_sent,y_tags,y_sent in zip(x,y_txt,y):
            if specific_label is not None and specific_label!=y_sent: continue
            if pos_or_neg=="pos" and y_sent=="O": continue
            elif pos_or_neg=="neg" and y_sent!="O": continue                
            if y_sent=="O":
                it_is_train=0
            else:
                it_is_train=it_is_train_proxy               
            tmp=tokenizer(split_sent,is_split_into_words=False)["input_ids"]
            tmp_x=[]
            tmp_attn=[]
            tmp_y=[]
            for i,chunk in enumerate(tmp):
                if for_protos and y_tags[i]=="O":
                    continue
                tmp_y.extend([y_tags[i]]*len(chunk))
                if y_tags[i]!="O":
                    mask=1
                else:
                    if it_is_train:
                        mask=0
                    else:
                        mask=1
                tmp_x.extend(chunk[1:-1])
                tmp_attn.extend([mask]*(len(chunk)-2))
            tmp_x.append(tokenizer.eos_token_id)
            tmp_x.insert(0,tokenizer.bos_token_id)
            tmp_attn.append(tmp_attn[-1])
            tmp_attn.insert(0,tmp_attn[0])
            self.x.append(tmp_x)
            self.attn_mask.append(tmp_attn)
            self.y_txt.append(tmp_y)
            self.y.append(1 if y_sent!="O" else 0)
            self.y_fine_int.append(self.labels_ids[y_sent])
        for tokid_sent in self.x:
            tokid_sent.extend([tokenizer.pad_token_id]*(fix_seq_len-len(tokid_sent)))
        for mask_vec in self.attn_mask:
            mask_vec.extend([0]*(fix_seq_len-len(mask_vec)))
#         self.y=[1 if i!="O" else 0 for i in y]
        if balance:
            num_pos=np.sum(self.y)
            assert num_pos<len(self.y_fine_int)//2
#             print(num_pos,len(self.y))
            
            pos_indices=np.random.choice([i for i in range(len(self.y)) if self.y[i]==1],
                                         size=len(self.y)-2*num_pos,replace=True)
            self.x.extend([self.x[i] for i in pos_indices])
            self.y.extend([1 for i in pos_indices])
            self.y_fine_int.extend([self.y_fine_int[i] for i in pos_indices])
            self.attn_mask.extend([self.attn_mask[i] for i in pos_indices])
#         print(np.sum(self.y),len(self.y))
        self.fix_seq_len=fix_seq_len
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

