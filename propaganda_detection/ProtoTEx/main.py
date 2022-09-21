import os
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/infonas/data/baekgupta/cache/'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2" 
from importlib import reload  
import numpy as np
import torch,time
from transformers import BartModel,BartConfig,BartForConditionalGeneration
from transformers import BartTokenizer
from tqdm.notebook import tqdm
import pathlib
from args import args

## Custom modules
from preprocess import make_dataset
from preprocess import make_bert_dataset,make_bert_testset
from preprocess import create_labels, labels_set 
from preprocess import BinaryClassDataset

from training import train_simple_ProtoTEx, train_simple_ProtoTEx_adv, train_ProtoTEx_w_neg

## Set cuda 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():    
    ## preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    ## https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html 

    train=make_dataset(pathlib.Path("../data/protechn_corpus_eval/train/"))
    val=make_dataset(pathlib.Path("../data/protechn_corpus_eval/dev/"))
    test=make_dataset(pathlib.Path("../data/protechn_corpus_eval/test/"))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_=make_bert_testset(train)
    val_=make_bert_testset(val)
    test_=make_bert_testset(test)
    # train_sents=[ " ".join(i) for d in train_[0] for i in d]
    train_sents=[ list(map(lambda x: x[1] if x[0]==0 else " "+x[1], enumerate(i))) for d in train_[0] for i in d]
    # val_sents=[ " ".join(i) for d in val_[0] for i in d]
    val_sents=[ list(map(lambda x: x[1] if x[0]==0 else " "+x[1], enumerate(i))) for d in val_[0] for i in d]
    # test_sents=[ " ".join(i) for d in test_[0] for i in d]
    test_sents=[ list(map(lambda x: x[1] if x[0]==0 else " "+x[1], enumerate(i))) for d in test_[0] for i in d]

    
    train_ls=create_labels(train_)
    val_ls=create_labels(val_)
    test_ls=create_labels(test_)
    train_y_txt=[ i for d in train_[1] for i in d]
    val_y_txt=[ i for d in val_[1] for i in d]
    test_y_txt=[ i for d in test_[1] for i in d]


    train_idx_bylabel={x: [i for i in range(len(train_ls)) if train_ls[i]==x] for x in labels_set} 
    val_idx_bylabel={x: [i for i in range(len(val_ls)) if val_ls[i]==x] for x in labels_set} 
    test_idx_bylabel={x: [i for i in range(len(test_ls)) if test_ls[i]==x] for x in labels_set} 

    if(args.tiny_sample):
        tiny_sample_n = 100
        train_sents = train_sents[0:tiny_sample_n]
        train_ls = train_ls[0:tiny_sample_n]
        train_y_txt = train_y_txt[0:tiny_sample_n]

        val_sents = val_sents[0:tiny_sample_n]
        val_ls = val_ls[0:tiny_sample_n]
        val_y_txt = val_y_txt[0:tiny_sample_n] 
        
        test_sents = test_sents[0:tiny_sample_n]
        test_ls = test_ls[0:tiny_sample_n]
        test_y_txt = test_y_txt[0:tiny_sample_n]

    train_dataset=BinaryClassDataset(train_sents,train_ls,train_y_txt,it_is_train=0,balance=True, tokenizer=tokenizer)
    val_dataset=BinaryClassDataset(val_sents,val_ls,val_y_txt,it_is_train=0, tokenizer=tokenizer)
    test_dataset=BinaryClassDataset(test_sents,test_ls,test_y_txt,it_is_train=0, tokenizer=tokenizer)
    train_dataset_eval=BinaryClassDataset(train_sents,train_ls,train_y_txt,it_is_train=0,balance=False, tokenizer=tokenizer)
    

    train_dl=torch.utils.data.DataLoader(train_dataset,batch_size=20,shuffle=True,
                                     collate_fn=train_dataset.collate_fn)
    val_dl=torch.utils.data.DataLoader(val_dataset,batch_size=128,shuffle=False,
                                     collate_fn=val_dataset.collate_fn)
    test_dl=torch.utils.data.DataLoader(test_dataset,batch_size=128,shuffle=False,
                                     collate_fn=test_dataset.collate_fn)
    train_dl_eval=torch.utils.data.DataLoader(train_dataset_eval,batch_size=20,shuffle=False,
                                     collate_fn=train_dataset_eval.collate_fn)


    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}, {1}".format(args.num_prototypes, args.num_pos_prototypes))
        train_ProtoTEx_w_neg(
            train_dl =  train_dl,
            val_dl = val_dl,
            test_dl = test_dl,
            num_prototypes=args.num_prototypes, 
            num_pos_prototypes=args.num_pos_prototypes
        )
    # SimpleProtoTEx can be trained in two different ways. In one case it is by reusing the ProtoTEx class definition 
    # and the other way is to use a dedicated SimpleProtoTEx class definition. Both of the implementations are available below. 
    # The dedicated SimpleProtoTEx class definition shall reproduce the results mentioned in the paper. 
    
    elif args.model == "SimpleProtoTExAdv":
        print("Use ProtoTEx Class definition for Simple ProtoTEx")
        train_simple_ProtoTEx_adv(
            train_dl = train_dl,
            val_dl = val_dl,
            test_dl = test_dl,
            train_dataset_len = len(train_dataset),
            num_prototypes=args.num_prototypes, 
            num_pos_prototypes=args.num_pos_prototypes
        ) 
    
    elif args.model == "SimpleProtoTEx":
        print("Dedicated simple prototex")
        train_simple_ProtoTEx(
             train_dl, 
             val_dl, 
             test_dl,
             train_dataset_len = len(train_dataset),
             modelname="0406_simpleprotobart_onlyclass_lp1_lp2_fntrained_20_train_nomask_protos",
             num_prototypes=args.num_prototypes 
        )


if __name__ == '__main__':
    main()