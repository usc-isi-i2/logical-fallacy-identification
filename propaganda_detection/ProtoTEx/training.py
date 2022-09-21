import numpy as np
import torch
from transformers.optimization import AdamW
from tqdm.notebook import tqdm


## Custom modules
from utils import EarlyStopping, print_logs, evaluate
from models import ProtoTEx, SimpleProtoTex

## Save paths
MODELPATH = "Models/"
LOGSPATH = "Logs/"

#### Training and eval functions

def train_simple_ProtoTEx(
    train_dl, 
    val_dl, 
    test_dl,
    num_prototypes,
    train_dataset_len,
    modelname="0406_simpleprotobart_onlyclass_lp1_lp2_fntrained_20_train_nomask_protos",
):
    torch.cuda.empty_cache()        
    model=SimpleProtoTex(num_prototypes=num_prototypes).cuda()
    torch.cuda.empty_cache()
    
    save_path=MODELPATH+modelname
    logs_path=LOGSPATH+modelname    

    optim=AdamW(model.parameters(),lr=3e-5,weight_decay=0.01,eps=1e-8)
    f=open(logs_path,"w")
    f.writelines([""])
    f.close()
    epoch=-1
    val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
    print_logs(logs_path,"VAL SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
    val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(train_dl,model)
    print_logs(logs_path,"TRAIN SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
    es=EarlyStopping(-np.inf,patience=7,path=save_path,save_epochwise=False)
    n_iters=500
    for epoch in range(n_iters):
        total_loss=0
        model.train()
        model.set_encoder_status(status=True)
        model.set_decoder_status(status=False)
        model.set_protos_status(status=True)
        model.set_classfn_status(status=True)
        classfn_loss,rc_loss,l_p1,l_p2,l_p3=[0]*5
        train_loader = tqdm(train_dl, total=len(train_dl), unit="batches",desc="training")
        for batch in train_loader:
            input_ids,attn_mask,y=batch
            classfn_out,loss=model(input_ids,attn_mask,y,use_decoder=0,use_classfn=1,
                                   use_rc=0,use_p1=1,use_p2=1,rc_loss_lamb=1.0,p1_lamb=1.0,
                                   p2_lamb=1.0)
            optim.zero_grad()
            loss[0].backward()
            optim.step()
            classfn_out=None
            loss=None
        total_loss=total_loss/train_dataset_len
        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(train_dl,model)
        print_logs(logs_path,"TRAIN SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        es.activate(mac_val_f1[0],mac_val_f1[1])
        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
        print_logs(logs_path,"VAL SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        es((mac_val_f1[1]+mac_val_f1[0])/2,epoch,model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"TEST SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        elif (epoch+1)%5==0:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"TEST SCORES (not the best ones)",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)


def train_simple_ProtoTEx_adv(
        train_dl, 
        val_dl, 
        test_dl,
        train_dataset_len,
        num_prototypes, 
        num_pos_prototypes,
        modelname="0406_simpleprotobart_onlyclass_lp1_lp2_fntrained_20_train_nomask_protos"
        ):
    torch.cuda.empty_cache()
    model=ProtoTEx(num_prototypes, num_pos_prototypes).cuda()
    model.set_prototypes(do_random=True)
    optim=AdamW(model.parameters(),lr=3e-5,weight_decay=0.01,eps=1e-8)
    save_path=MODELPATH+modelname
    logs_path=LOGSPATH+modelname
    f=open(logs_path,"w")
    
    f.writelines([""])
    f.close()
    epoch=-1
    val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
    print_logs(logs_path,"VAL SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
    val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(train_dl,model)
    print_logs(logs_path,"TRAIN SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
    es=EarlyStopping(-np.inf,patience=7,path=save_path,save_epochwise=False)
    n_iters=500
    for epoch in range(n_iters):
        total_loss=0
        model.train()
        model.set_encoder_status(status=False)
        model.set_decoder_status(status=False)
        model.set_protos_status(status=True)
        model.set_classfn_status(status=True)
        classfn_loss,rc_loss,l_p1,l_p2,l_p3=[0]*5
        train_loader = tqdm(train_dl, total=len(train_dl), unit="batches",desc="training")
        for batch in train_loader:
            input_ids,attn_mask,y=batch
            classfn_out,loss=model(input_ids,attn_mask,y,use_decoder=0,use_classfn=1,
                                use_rc=0,use_p1=1,use_p2=1,use_p3=0,rc_loss_lamb=1.0,p1_lamb=1.0,
                                p2_lamb=1.0,p3_lamb=0)
            total_loss+=loss[0].detach().item()
            classfn_loss+=loss[1].detach().item()
            rc_loss+=loss[2].detach().item()
            l_p1+=loss[3].detach().item()
            l_p2+=loss[4].detach().item()
            l_p3+=loss[5].detach().item()
            optim.zero_grad()
    #             loss=loss/len(batch)
            loss[0].backward()
            optim.step()
            classfn_out=None
            loss=None
    #             torch.cuda.empty_cache()
        print(classfn_loss,rc_loss,l_p1,l_p2,l_p3)
        # total_loss=total_loss/len(train_dataset)
        total_loss=total_loss/train_dataset_len
        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(train_dl,model)
        print_logs(logs_path,"TRAIN SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)

        es.activate(mac_val_f1[0],mac_val_f1[1])
        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
        print_logs(logs_path,"VAL SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        
    #     es((mac_val_f1[1]+mac_val_f1[0])/2,epoch,model)
        es(mac_val_f1[1],epoch,model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"TEST SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        elif (epoch+1)%5==0:
            # print_logs(logs_path,"Early Stopping VAL SCORES (not the best)",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)

            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"Early Stopping TEST SCORES (not the best ones)",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)


def train_ProtoTEx_w_neg(train_dl,
                        val_dl,
                        test_dl,
                        num_prototypes, 
                        num_pos_prototypes,
                        modelname="0408_NegProtoBart_protos_xavier_large_bs20_20_woRat_noReco_g2d_nobias_nodrop_cu1_PosUp_normed"
                        ):
    torch.cuda.empty_cache()
    model=ProtoTEx(num_prototypes=num_prototypes, 
                    num_pos_prototypes=num_pos_prototypes,
                    bias=False, 
                    dropout=False, 
                    special_classfn=True, # special_classfn=False, ## apply dropouonly on bias 
                    p=1, #p=0.75,
                    batchnormlp1=True
                    ).cuda()
    model.set_prototypes(do_random=True)
    
    optim=AdamW(model.parameters(),lr=3e-5,weight_decay=0.01,eps=1e-8)
    # optim=AdamW([{'params':model.bart_model.parameters()},
    #              {'params':list(model.classfn_model.parameters())+
    #                         [model.pos_prototypes]+
    #                         [model.neg_prototypes],'lr':3e-4}],lr=3e-5,weight_decay=0.01,eps=1e-8)
    # modelname="NegProtoBart_Ratprotos_nice_large_bs20_20_woRat_noReco_g2d1_bias_drop_cu0_dectrained"
    save_path=MODELPATH+modelname
    logs_path=LOGSPATH+modelname
    f=open(logs_path,"w")
    f.writelines([""])
    f.close()
    val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
    epoch=-1
    print_logs(logs_path,"VAL SCORES",epoch,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
    es=EarlyStopping(-np.inf,patience=7,path=save_path,save_epochwise=False)
    n_iters=1000
    gamma=2
    delta=1
    kappa=1
    p1_lamb=0.9
    p2_lamb=0.9
    p3_lamb=0.9
    for iter_ in range(n_iters):
        total_loss = 0
        """
        During Delta, We want decoder to become better at decoding the trained encoder
        and Prototypes to become closer to some encoded representation. And that's why it makes 
        sense to use l_p1 loss and not l_p2 loss.
        losses- rc_loss, l_p1 loss
        trainable- decoder and prototypes
        details- makes pos_prototypes closer to pos_egs and neg_protos closer to neg_egs 
        """
        model.train()
        model.set_encoder_status(status=False)
        model.set_decoder_status(status=False)
        model.set_protos_status(status=True)
        model.set_classfn_status(status=False)
        model.set_shared_status(status=True)
    #     print("\n after gamma")
    #     for n,w in model.named_parameters():
    #         if w.requires_grad: print(n)
        for epoch in range(delta):
            train_loader = tqdm(train_dl, total=len(train_dl), unit="batches", desc="delta training")
            for batch in train_loader:
                input_ids, attn_mask, y = batch
    #             print(y)
                classfn_out, loss = model(input_ids, attn_mask, y, use_decoder=0, use_classfn=0,
                                        use_rc=0, use_p1=1, use_p2=0, use_p3=0,
                                        rc_loss_lamb=1.0, p1_lamb=p1_lamb, p2_lamb=p2_lamb,
                                        p3_lamb=p3_lamb,distmask_lp1=1,distmask_lp2=1,
                                        random_mask_for_distanceMat=None)
                # total_loss += loss[0].detach().item()
                optim.zero_grad()
                loss[0].backward()
                optim.step()
        """
        During gamma, we only want to improve the classification performance. Therefore we will
        improve encoder to become closer to the prototypes, at the same time also improving
        the classification accuracy. That's why encoder and classification layer must be trainabl
        together without segrregating pos and neg examples.
        Only Encoder and Classfn are trainable
        """
        model.train()
        model.set_encoder_status(status=True)
        model.set_decoder_status(status=False)
        model.set_protos_status(status=False)
        model.set_classfn_status(status=True)
        model.set_shared_status(status=True)
    #     print("\n after gamma")
    #     for n,w in model.named_parameters():
    #         if w.requires_grad: print(n)
        for epoch in range(gamma):
            train_loader = tqdm(train_dl, total=len(train_dl), unit="batches", desc="gamma training")
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                classfn_out, loss = model(input_ids, attn_mask, y, use_decoder=0, use_classfn=1,
                                        use_rc=0, use_p1=0, use_p2=1,
                                        rc_loss_lamb=1., p1_lamb=p1_lamb,p2_lamb=p2_lamb,
                                        distmask_lp1 = 1, distmask_lp2 = 1)
                optim.zero_grad()
                loss[0].backward()
                optim.step()
    #     """
    #     During Kappa, we only want to improve the reconstruction perf.
    #     Only Decoder is  trainable
    #     """
    #     model.train()
    #     model.set_encoder_status(status=False)
    #     model.set_decoder_status(status=True)
    #     model.set_protos_status(status=False)
    #     model.set_classfn_status(status=False)
    #     for epoch in range(kappa):
    #         train_loader = tqdm(train_dl, total=len(train_dl), unit="batches", desc="delta training")
    #         for batch in train_loader:
    #             input_ids, attn_mask, y = batch
    # #             print(y)
    #             classfn_out, loss = model(input_ids, attn_mask, y, use_decoder=1, use_classfn=0,
    #                                       use_rc=1, use_p1=0, use_p2=0, use_p3=0,
    #                                       rc_loss_lamb=1.0, p1_lamb=p1_lamb, p2_lamb=p2_lamb,
    #                                       p3_lamb=p3_lamb,distmask_lp1=1,distmask_lp2=1,
    #                                       random_mask_for_distanceMat=None)
    #             # total_loss += loss[0].detach().item()
    #             optim.zero_grad()
    #             loss[0].backward()
    #             optim.step()

        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(train_dl,model)
        print_logs(logs_path,"TRAIN SCORES",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)
        es.activate(mac_val_f1[0],mac_val_f1[1])
        val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(val_dl,model)
        print_logs(logs_path,"VAL SCORES",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)        

        es(0.5*(mac_val_f1[1]+mac_val_f1[0]),epoch,model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"TEST SCORES",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)

        elif (iter_+1)%5==0:
            
            # print_logs(logs_path,"Early Stopping VAL SCORES (not the best ones)",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)        
            
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1=evaluate(test_dl,model)
            print_logs(logs_path,"Early Stopping TEST SCORES (not the best ones)",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1,mic_val_prec,mic_val_rec,mic_val_f1)

