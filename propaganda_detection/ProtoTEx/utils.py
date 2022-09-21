from sklearn.metrics import precision_recall_fscore_support
from transformers.modeling_outputs import BaseModelOutput
import torch 
from tqdm import tqdm
import numpy as np

## Custom modules
from preprocess import BinaryClassDataset, labels_set

def print_logs(file,info,epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1):
    logs=[]
    s=" ".join((info+" epoch",str(epoch),"Total loss %.4f"%(val_loss),"\n"))
    logs.append(s)
    print(s)
    s=" ".join((info+" epoch",str(epoch),"Prec",str(mac_val_prec),"\n"))
    logs.append(s)
    print(s)
    s=" ".join((info+" epoch",str(epoch),"Recall",str(mac_val_rec),"\n"))
    logs.append(s)
    print(s)
    s=" ".join((info+" epoch",str(epoch),"F1",str(mac_val_f1),"\n"))
    logs.append(s)
    print(s)
#     print("epoch",epoch,"MICRO val precision %.4f, recall %.4f, f1 %.4f,"%(mic_val_prec,mic_val_rec,mic_val_f1))
    print() 
    logs.append("\n")
    f=open(file,"a")
    f.writelines(logs)
    f.close()


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, score_at_min1=0,patience=100, verbose=False, delta=0, path='checkpoint.pt',
                 trace_func=print,save_epochwise=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = score_at_min1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.state_dict_list=[None]*patience
        self.improved=0
        self.stop_update=0
        self.save_model_counter=0
        self.save_epochwise=save_epochwise
        self.times_improved=0
        self.activated=False
    def activate(self,s1,s2):
        if not self.activated and s1>0 and s2>0: self.activated=True
    def __call__(self, score, epoch,model):
        if not self.activated: return None
        self.save_model_counter = (self.save_model_counter + 1) % 4
        if not self.stop_update:
            if self.verbose:
                self.trace_func(f'\033[91m The val score  of epoch {epoch} is {score:.4f} \033[0m')
            if score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'\033[93m EarlyStopping counter: {self.counter} out of {self.patience} \033[0m')
                if self.counter >= self.patience:
                    self.early_stop = True
                self.improved=0
            else:
                self.save_checkpoint(score, model,epoch)
                self.best_score = score
                self.counter = 0
                self.improved=1
        else:
            self.improved=0 #not needed though

    def save_checkpoint(self, score, model, epoch):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        self.times_improved+=1
        self.trace_func(f'\033[92m Validation score improved ({self.best_score:.4f} --> {score:.4f}). \033[0m')
        if self.save_epochwise:
            path=self.path+"_"+str(self.times_improved)+"_"+str(epoch)
        else:
            path=self.path
        torch.save(model.state_dict(), path)

def evaluate(dl,model_new=None,path=None,modelclass=None):
    assert (model_new is not None) ^ (path is not None)
    if path is not None:
        model_new=modelclass().cuda()
        model_new.load_state_dict(torch.load(path))
    loader = tqdm(dl, total=len(dl), unit="batches")
    total_len=0
    model_new.eval()    
    with torch.no_grad():
        total_loss=0
        tts=0
        y_pred=[]
        y_true=[]
        for batch in loader:
            input_ids,attn_mask,y=batch
            classfn_out,loss=model_new(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
#             print(classfn_out.detach().cpu())
            if classfn_out.ndim==1:
                predict=torch.zeros_like(y)
                predict[classfn_out>0]=1
            else:
                predict=torch.argmax(classfn_out,dim=1)
                
            y_pred.append(predict.cpu().numpy())
#             y_pred.append(torch.zeros_like(y).numpy())
            y_true.append(y.cpu().numpy())
            total_loss+=(len(input_ids)*loss[0].item())
            total_len+=len(input_ids)
#             torch.cuda.empty_cache()            
        total_loss=total_loss/total_len
        mac_prec,mac_recall,mac_f1_score,_=precision_recall_fscore_support(np.concatenate(y_true),np.concatenate(y_pred),labels=[0,1])
#         mic_prec,mic_recall,mic_f1_score,_=precision_recall_fscore_support(np.concatenate(y_true),np.concatenate(y_pred),labels=[0,1])
        mic_prec,mic_recall,mic_f1_score,_=0,0,0,0

    return total_loss,mac_prec,mac_recall,mac_f1_score,mic_prec,mic_recall,mic_f1_score


#### Functions for analyzing prototypes 

def get_best_k_protos_for_batch(test_sents, test_ls, test_y_txt, specific_label, tokenizer, model_new=None,model_path=None,
                                model_class=None,topk=None,do_all=False):
    """
        get the best k protos for that a fraction of test data where each element has a specific true label.
        the "best" is in the senese that it has (or is one of those who has) the minimal distance 
        from the encoded representation of the sentence.
    """
    assert (model_new is not None) ^ (model_path is not None)
    if model_new is None:
        print("creating new model")
        model_new=model_class().cuda()
        model_new.load_state_dict(torch.load(model_path))
    dataset=BinaryClassDataset(test_sents,test_ls,test_y_txt,tokenizer,it_is_train=0,specific_label=specific_label)
    dl=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,
                                     collate_fn=dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    with torch.no_grad():
        all_protos=torch.cat((model_new.pos_prototypes,model_new.neg_prototypes),
                             dim=0).view(model_new.num_protos,-1)
        best_protos=[]
        best_protos_dists=[]
        indices=[]
        for batch in loader:
            input_ids,attn_mask,y=batch
#             print(y)
            batch_size=input_ids.size(0)
            last_hidden_state=model_new.bart_model.base_model.encoder(input_ids.cuda(),attn_mask.cuda(),
                                                                 output_attentions=False,
                                                                 output_hidden_states=False).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = model_new.one_by_sqrt_bartoutdim * torch.cdist(last_hidden_state.view(batch_size, -1),
                                                                              all_protos.view(model_new.num_protos, -1))
            else:
                input_for_classfn = torch.cdist(last_hidden_state.view(batch_size, -1),
                                                all_protos.view(model_new.num_protos, -1))
                input_for_classfn= torch.nn.functional.instance_norm(
                    input_for_classfn.view(batch_size,1,model_new.num_protos)).view(batch_size,
                                                                               model_new.num_protos)
            predicted=torch.argmax(model_new.classfn_model(input_for_classfn).view(batch_size, 2),dim=1)
            if do_all:
                temp=torch.topk(input_for_classfn,dim=1,
                                k=topk,largest=False)
            else:
                concerned_idxs=torch.nonzero((predicted==y.cuda())).view(-1)
                temp=torch.topk(input_for_classfn[concerned_idxs],dim=1,
                                k=topk,largest=False)
            best_protos.append(temp[1].cpu())
            best_protos_dists.append((temp[0]*torch.sqrt(torch.tensor(model_new.bart_out_dim).float())).cpu())
#             best_protos.append((torch.topk(input_for_classfn,dim=1,
#                                               k=topk,largest=False)[1]).cpu())
        best_protos=torch.cat(best_protos,dim=0)
    return best_protos,torch.cat(best_protos_dists,dim=0)

# Obtain the prototype distances
def get_cooccurence_matrix(model_new=None,model_path=None,model_class=None,
                          specific_labels=None,topk=3):
    """
        get the aggrgated cooccurence matrix of shape num_labels x num_protos
    """
    assert (model_new is not None) ^ (model_path is not None)
    if model_new is None:
        print("creating new model")
        model_new=model_class().cuda()
        model_new.load_state_dict(torch.load(model_path))
    labels_list=[label for label in labels_set]
    data=np.zeros((len(labels_list),model_new.num_protos))
    if specific_labels is not None: labels_list=specific_labels
    for i,label in enumerate(labels_list):
        print(label)
        temp=get_best_k_protos_for_batch(label,model_new=model_new,topk=topk)[0].view(-1).numpy()
        best_protos_per_label=dict(zip(*np.unique(temp,
                                        return_counts=True)))
#         print()
        for protos_idx in best_protos_per_label:
            data[i][protos_idx]=best_protos_per_label[protos_idx]
    return data

def get_bestk_train_data_for_every_proto(train_dataset_eval, model_new=None,top_k=3,return_distances=True):
    """
        for every prototype find out k best similar training examples
    """
    batch_size=128
    dl=torch.utils.data.DataLoader(train_dataset_eval,batch_size=batch_size,shuffle=False,
                                     collate_fn=train_dataset_eval.collate_fn)
#     dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
#                                      collate_fn=test_dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
#     model_new=model_new.cpu()
    count=0
    with torch.no_grad():
        best_train_egs=[]
        best_train_egs_values=[]
        all_distances=torch.tensor([])
        predict_all=torch.tensor([])
        true_all=torch.tensor([])
        all_protos=torch.cat((model_new.pos_prototypes,model_new.neg_prototypes),
                             dim=0).view(model_new.num_protos,-1)
        for batch in loader:
            input_ids,attn_mask,y=batch
            batch_size=input_ids.size(0)
            last_hidden_state=model_new.bart_model.base_model.encoder(input_ids.cuda(),attn_mask.cuda(),
#             last_hidden_state=model_new.bart_model.base_model.encoder(input_ids,attn_mask,
                                                                 output_attentions=False,
                                                                 output_hidden_states=False).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = model_new.one_by_sqrt_bartoutdim * torch.cdist(last_hidden_state.view(batch_size, -1),
                                                                              all_protos.view(model_new.num_protos, -1))
            else:
                input_for_classfn = torch.cdist(last_hidden_state.view(batch_size, -1),
                                                all_protos.view(model_new.num_protos, -1))
                input_for_classfn= torch.nn.functional.instance_norm(
                    input_for_classfn.view(batch_size,1,model_new.num_protos)).view(batch_size,
                                                                               model_new.num_protos)
            predicted=torch.argmax(model_new.classfn_model(input_for_classfn).view(batch_size, 2),dim=1)
            concerned_idxs=torch.nonzero((predicted==y.cuda())).view(-1)
#             concerned_idxs=torch.nonzero((predicted==y)).view(-1)
            input_for_classfn=input_for_classfn[concerned_idxs]*torch.sqrt(torch.tensor(model_new.bart_out_dim).float())
#             predict_all=torch.cat((predict_all,predicted.cpu()),dim=0)
#             true_all=torch.cat((true_all,y.cpu()),dim=0)
            if top_k is None:
                all_distances=torch.cat((all_distances,input_for_classfn.cpu()),dim=0)
            else:
                best=torch.topk(input_for_classfn,dim=0,k=top_k,largest=False)
                best_train_egs.append((best[1]+count*batch_size))
                count+=1
                best_train_egs_values.append(best[0])
    if top_k is None:
        return torch.cat((true_all.view(-1,1),predict_all.view(-1,1),all_distances),dim=1)
    else:
        best_train_egs=torch.cat(best_train_egs,dim=0)
        best_train_egs_values=torch.cat(best_train_egs_values,dim=0)
        temp=torch.topk(best_train_egs_values,dim=0,k=top_k,largest=False)
        topk_idxs=temp[1]
        final_concerned_idxs=[]
        for i in range(best_train_egs.size(1)):
            concerned_idxs=best_train_egs[topk_idxs[:,i],i]
            final_concerned_idxs.append(concerned_idxs)
#         true_all=torch.cat(true_all,dim=0)
#         predict_all=torch.cat(predict_all,dim=0)
        return torch.stack(final_concerned_idxs,dim=0).cpu().numpy(),temp[0].cpu().numpy()

def get_distances_for_rdm(train_dataset_eval, model_new=None,return_distances=True):
    """
        for every prototype find out k best similar training examples
    """
    batch_size=30
    dl=torch.utils.data.DataLoader(train_dataset_eval,batch_size=batch_size,shuffle=True,
                                     collate_fn=train_dataset_eval.collate_fn)
#     dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
#                                      collate_fn=test_dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    count=0
    with torch.no_grad():
        best_train_egs=[]
        best_train_egs_values=[]
        all_distances=torch.tensor([])
        predict_all=torch.tensor([])
        true_all=torch.tensor([])
        all_protos=torch.cat((model_new.pos_prototypes,model_new.neg_prototypes),
                             dim=0).view(model_new.num_protos,-1)
        for batch in loader:
            input_ids,attn_mask,y=batch
            batch_size=input_ids.size(0)
            last_hidden_state=model_new.bart_model.base_model.encoder(input_ids.cuda(),attn_mask.cuda(),
#             last_hidden_state=model_new.bart_model.base_model.encoder(input_ids,attn_mask,
                                                                 output_attentions=False,
                                                                 output_hidden_states=False).last_hidden_state
            input_for_classfn=model_new.one_by_sqrt_bartoutdim* torch.cdist(last_hidden_state.view(batch_size,-1),
                                          all_protos)
            predicted=torch.argmax(model_new.classfn_model(input_for_classfn).view(batch_size, 2),dim=1)
            concerned_idxs=torch.nonzero(torch.logical_and(predicted==y.cuda(),y.cuda()==1)).view(-1)
#             concerned_idxs=torch.nonzero((predicted==y)).view(-1)
            input_for_classfn=input_for_classfn[concerned_idxs]
            print(torch.sort(input_for_classfn,descending=False,dim=1)[0].cpu())
            break
        return    

def print_protos(tokenizer, train_ls, which_protos, protos_train_table):
    for i in which_protos:
        concerned_idxs=protos_train_table[i]
        sents=tokenizer.batch_decode([train_dataset.x[j] for j in concerned_idxs],skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        for (sent,y) in zip(sents,[train_ls[j] for j in concerned_idxs]):
            print(sent,y)
        print()

def best_protos_for_test(test_dataset, model_new=None):
    batch_size=60
    dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,
                                     collate_fn=test_dataset.collate_fn)
#     loader = tqdm(dl, total=len(dl), unit="batches")
    all_protos=torch.cat((model_new.pos_prototypes,model_new.neg_prototypes),dim=0)
    input_ids,attn_mask,y=next(iter(dl))
    with torch.no_grad():
        last_hidden_state=model_new.bart_model.base_model.encoder(input_ids.cuda(),attn_mask.cuda(),
                                                             output_attentions=False,
                                                             output_hidden_states=False).last_hidden_state
        input_for_classfn=torch.cdist(last_hidden_state.view(batch_size,-1),
                                      all_protos.view(model_new.num_protos,-1))
        predicted=torch.argmax(model_new.classfn_model(input_for_classfn),dim=1)
        proper_idxs_pos=(torch.nonzero(torch.logical_and(predicted==y,y==1)).view(-1))[:15]
        proper_idxs_neg=(torch.nonzero(torch.logical_and(predicted==y,y==0)).view(-1))[:15]
        pos_best_protos=torch.topk(input_for_classfn[proper_idxs_pos],dim=1,k=top_k,largest=False)[1]
        neg_best_protos=torch.topk(input_for_classfn[proper_idxs_neg],dim=1,k=top_k,largest=False)[1]
    return (input_ids[proper_idxs_pos],pos_best_protos),(input_ids[proper_idxs_pos],neg_best_protos)
    
def get_distance_div(train_sents, train_ls, train_y_txt, model_new=None):
    labels_list=[label for label in labels_set]
#     labels_list=labels_list[:4]
    input_ids,attn_mask,y=[],[],[]
    samples_from_each_label=3
    num_labels_actual=0
    for specific_label in labels_list:
#         dataset=BinaryClassDataset(test_sents,test_ls,test_y_txt,it_is_train=0,specific_label=specific_label)
        dataset=BinaryClassDataset(train_sents,train_ls,train_y_txt,it_is_train=0,specific_label=specific_label)
        if len(dataset)<samples_from_each_label: continue
        dl=torch.utils.data.DataLoader(dataset,batch_size=samples_from_each_label,shuffle=True,collate_fn=dataset.collate_fn)
        a,b,c=next(iter(dl))
        input_ids.append(a)
        attn_mask.append(b)
        y.append(c)
        num_labels_actual+=1
        print(specific_label)
    input_ids=torch.cat(input_ids,dim=0)
    attn_mask=torch.cat(attn_mask,dim=0)
    y=torch.cat(y,dim=0)
    model_new.eval()
    with torch.no_grad():
        all_protos=torch.cat((model_new.pos_prototypes,model_new.neg_prototypes),
                             dim=0).view(model_new.num_protos,-1)
        batch_size=input_ids.size(0)
        last_hidden_state=model_new.bart_model.base_model.encoder(input_ids.cuda(),attn_mask.cuda(),
                                                             output_attentions=False,
                                                             output_hidden_states=False).last_hidden_state
        all_distances=model_new.one_by_sqrt_bartoutdim*torch.cdist(last_hidden_state.view(batch_size,-1),
                                      all_protos)
        all_distances_normized=torch.nn.functional.softmax(1./all_distances,dim=1)
        all_kldiv=[]
        for i in range(num_labels_actual):
            p=all_distances_normized[i*samples_from_each_label:(i+1)*samples_from_each_label]
            p_repeated=p.repeat(num_labels_actual,1)
#             print(all_distances_normized.size(),p_repeated.size())
            kldiv=torch.mean(torch.nn.KLDivLoss(reduction="none")(all_distances_normized,p_repeated),dim=-1)
            kldiv_by_class=torch.mean(kldiv.view(-1,samples_from_each_label),dim=-1)
            all_kldiv.append(kldiv_by_class)
    #             predicted=torch.argmax(model_new.classfn_model(input_for_classfn),dim=1)
    #             concerned_idxs=torch.nonzero((predicted==y.cuda())).view(-1)
    return torch.stack(all_kldiv,dim=0)
