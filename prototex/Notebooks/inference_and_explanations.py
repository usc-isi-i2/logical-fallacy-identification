import os 
from IPython import embed
from importlib import reload  
import numpy as np
import torch,time
from transformers import BartModel,BartConfig,BartForConditionalGeneration,BartForCausalLM, BartTokenizer
from tqdm.notebook import tqdm
from torch import nn
import sys

MOD_FOLDER = '../'
# setting path to enable import from the parent directory
sys.path.append(MOD_FOLDER)
print(sys.path)
## Load all the functions for analyzing prototypes
from utils import *
from models import ProtoTEx

num_prototypes = 50
num_pos_prototypes = 49
model = ProtoTEx(num_prototypes, 
                 num_pos_prototypes,
                 n_classes=14,
                 bias=False, 
                 dropout=False, 
                 special_classfn=True, # special_classfn=False, ## apply dropout only on bias 
                 p=1, #p=0.75,
                 batchnormlp1=True)


model_path = "../Models/curr_lf_fine_updated_aug_with_none_nli_prototex"

print(f"Loading model checkpoint: {model_path}")
pretrained_dict = torch.load(model_path)
# Fiter out unneccessary keys
model_dict = model.state_dict()
filtered_dict = {}
for k, v in pretrained_dict.items():
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_dict[k] = v
    else:
        print(f"Skipping weights for: {k}")

model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

if model.num_neg_protos > 0:
    all_protos = torch.cat((model.neg_prototypes, model.pos_prototypes), dim=0)
else:
    all_protos = model.pos_prototypes
torch.save(all_protos, "all_protos.pt")

device = torch.device('cuda')
model.to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

## Load all the data classes
from preprocess import *
import pandas as pd

## Load data
train_df = pd.read_csv("../data/finegrained_with_none/train.csv")
dev_df = pd.read_csv("../data/finegrained_with_none/val.csv")
test_df = pd.read_csv("../data/finegrained_with_none/test.csv")

train_sentences = train_df['text'].tolist()
dev_sentences = dev_df['text'].tolist()
test_sentences = test_df['text'].tolist()

train_labels = train_df['label'].tolist()
dev_labels = dev_df['label'].tolist()
test_labels = test_df['label'].tolist()


train_dataset = CustomNonBinaryClassDataset(
    sentences = train_sentences,
    labels = train_labels,
    tokenizer = tokenizer
)
dev_dataset = CustomNonBinaryClassDataset(
    sentences = dev_sentences,
    labels = dev_labels,
    tokenizer=tokenizer
)
test_dataset = CustomNonBinaryClassDataset(
    sentences = test_sentences,
    labels = test_labels,
    tokenizer = tokenizer
)

train_dl=torch.utils.data.DataLoader(train_dataset,batch_size=20,shuffle=False,
                                 collate_fn=train_dataset.collate_fn)
val_dl=torch.utils.data.DataLoader(dev_dataset,batch_size=20,shuffle=False,
                                 collate_fn=dev_dataset.collate_fn)
test_dl=torch.utils.data.DataLoader(test_dataset,batch_size=20,shuffle=False,
                                 collate_fn=test_dataset.collate_fn)

test_sents = test_sentences

best_protos_per_testeg = get_best_k_protos_for_batch(
    dataset = test_dataset,
    specific_label=None, 
    model_new=model, 
    tokenizer=tokenizer, 
    topk= 5, 
    do_all=True
)
best_protos_per_traineg = get_best_k_protos_for_batch(
    dataset = train_dataset,
    specific_label=None, 
    model_new=model, 
    tokenizer=tokenizer, 
    topk= 5, 
    do_all=True
)
bestk_train_data_per_proto=get_bestk_train_data_for_every_proto(train_dataset, 
                                                   model_new=model, top_k=5)


joblib.dump(bestk_train_data_per_proto, "bestk_train_data_per_proto.joblib")
joblib.dump(best_protos_per_testeg, "best_protos_per_testeg.joblib")
joblib.dump(best_protos_per_traineg, "best_protos_per_traineg.joblib")
if model.num_neg_protos > 0:
    all_protos = torch.cat((model.neg_prototypes, model.pos_prototypes), dim=0)
else:
    all_protos = model.pos_prototypes
torch.save(all_protos, "all_protos.pt")

print_protos(
    train_dataset = train_dataset, 
    tokenizer = tokenizer, 
    train_ls = train_labels, 
    which_protos=list(range(num_prototypes)), 
    protos_train_table=bestk_train_data_per_proto[0]
)

# train_sents_joined = train_sentences
# test_sents_joined = test_sentences

"""
distances generation
test true labels and pred labels 
"""
# loader = tqdm(test_dl, total=len(test_dl), unit="batches")
# model.eval()    
# with torch.no_grad():
#     test_true=[]
#     test_pred=[]
#     for batch in loader:
#         input_ids,attn_mask,y=batch
#         classfn_out,_=model(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
#         predict=torch.argmax(classfn_out,dim=1)
# #         correct_idxs.append(torch.nonzero((predicted==y.cuda())).view(-1)
#         test_pred.append(predict.cpu().numpy())
#         test_true.append(y.cpu().numpy())
# test_true=np.concatenate(test_true)
# test_pred=np.concatenate(test_pred)

"""
distances generation
csv generation
"""
# import csv

# fields = ["S.No.", "Test Sentence","Predicted","Actual","Actual Prop or NonProp"]
# num_protos_per_test=5
# num_train_per_proto=5
# for i in range(num_protos_per_test):
#     for j in range(num_train_per_proto):
#         fields.append(f"Prototype_{i}_wieght0")
#         fields.append(f"Prototype_{i}_wieght1")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_actuallabel")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_distance")
        
# filename = f"{model_path[len('Models/'):]}_nearest.csv"
# weights=model.classfn_model.weight.detach().cpu().numpy()
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(fields)
#     for i in range(len(test_sents_joined)):
# #     for i in range(100):
#         row=[i,test_sents_joined[i],test_pred[i],test_labels[i],test_true[i]]
#         for j in range(num_protos_per_test):
#             proto_idx=best_protos_per_testeg[0][i][j]
#             for k in range(num_train_per_proto):
# #                 print(i,j,k)
#                 row.append(weights[0][proto_idx])
#                 row.append(weights[1][proto_idx])
#                 row.append(train_sents_joined[bestk_train_data_per_proto[0][proto_idx][k]])
#                 row.append(train_labels[bestk_train_data_per_proto[0][proto_idx][k]])
#                 row.append(bestk_train_data_per_proto[1][k][proto_idx])

#         csvwriter.writerow(row)

