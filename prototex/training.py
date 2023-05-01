import numpy as np
import torch
from transformers.optimization import AdamW
from tqdm.notebook import tqdm
from args import args
import wandb

## Custom modules
from utils import EarlyStopping, print_logs, evaluate
from models import ProtoTEx, ProtoTEx_roberta, ProtoTEx_electra

## Save paths
MODELPATH = "Models/"
LOGSPATH = "Logs/"

#### Training and eval functions

import torch as th
import math


class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print("Need scikit-learn for this functionality")
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def train_ProtoTEx_w_neg(
    train_dl,
    val_dl,
    test_dl,
    num_prototypes,
    num_pos_prototypes,
    class_weights=None,
    modelname="0408_NegProtoBart_protos_xavier_large_bs20_20_woRat_noReco_g2d_nobias_nodrop_cu1_PosUp_normed",
    model_checkpoint=None,
):
    torch.cuda.empty_cache()
    model = ProtoTEx(
        num_prototypes=num_prototypes,
        num_pos_prototypes=num_pos_prototypes,
        class_weights=class_weights,
        bias=False,
        dropout=False,
        special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
        p=1,  # p=0.75,
        batchnormlp1=True,
    ).cuda()

    sampler = StratifiedSampler(
        class_vector=torch.LongTensor(train_dl.dataset.y),
        batch_size=args.num_prototypes,
    )
    random_data_loader = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=args.num_prototypes,
            collate_fn=train_dl.dataset.collate_fn,
            sampler=sampler,
        )

    if model_checkpoint:
        print(f"Loading model checkpoint: {model_checkpoint}")
        pretrained_dict = torch.load(model_checkpoint)
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
    else:
        # TODO: Try getting the average of a first few batches
        batch = next(iter(random_data_loader))
        input_ids, attn_mask, y = batch
        model.set_prototypes(
            input_ids_pos_rdm=input_ids, attn_mask_pos_rdm=attn_mask, do_random=True
        )

    # Track all model parameters
    wandb.watch(
        models=model,
        criterion=model.loss_fn,
        log="all",
        log_freq=len(random_data_loader),
    )

    optim = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, eps=1e-8)

    save_path = MODELPATH + modelname
    logs_path = LOGSPATH + modelname
    f = open(logs_path, "w")
    f.writelines([""])
    f.close()
    val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(val_dl, model)
    epoch = -1
    print_logs(
        logs_path,
        "VAL SCORES",
        epoch,
        val_loss,
        mac_val_prec,
        mac_val_rec,
        mac_val_f1,
        accuracy,
    )
    es = EarlyStopping(-np.inf, patience=10, path=save_path, save_epochwise=False)
    n_iters = 1000
    gamma = 2
    delta = 1
    kappa = 1
    p1_lamb = 0.9
    p2_lamb = 0.9
    p3_lamb = 0.9
    for iter_ in range(n_iters):
        wandb.log({"epoch": iter_})
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
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="delta training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                #             print(y)
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=0,
                    use_rc=0,
                    use_p1=1,
                    use_p2=0,
                    use_p3=0,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    p3_lamb=p3_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                    random_mask_for_distanceMat=None,
                )
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
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="gamma training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=1,
                    use_rc=0,
                    use_p1=0,
                    use_p2=1,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                )
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

        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            train_dl, model
        )
        print_logs(
            logs_path,
            "TRAIN SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Train epoch": iter_,
                "Train loss": val_loss,
                "Train Precision": mac_val_prec,
                "Train Recall": mac_val_rec, 
                "Train Accuracy": accuracy,
                "Train F1 score": mac_val_f1,
            }
        )
        es.activate(mac_val_f1)
        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            val_dl, model
        )
        print_logs(
            logs_path,
            "VAL SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Val epoch": iter_,
                "Val loss": val_loss,
                "Val Precision": mac_val_prec,
                "Val Recall": mac_val_rec, 
                "Val Accuracy": accuracy,
                "Val F1 score": mac_val_f1,
            }
        )

        es(np.mean(mac_val_f1), epoch, model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "TEST SCORES",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )
            wandb.log(
            {
                "Test epoch": iter_,
                "Test loss": val_loss,
                "Test Precision": mac_val_prec,
                "Test Recall": mac_val_rec, 
                "Test Accuracy": accuracy,
                "Test F1 score": mac_val_f1,
            }
        )

        elif (iter_ + 1) % 5 == 0:

            # print_logs(logs_path,"Early Stopping VAL SCORES (not the best ones)",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1)

            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "Early Stopping TEST SCORES (not the best ones)",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )


def train_ProtoTEx_w_neg_roberta(
    train_dl,
    val_dl,
    test_dl,
    num_prototypes,
    num_pos_prototypes,
    class_weights=None,
    modelname="0408_NegProtoBart_protos_xavier_large_bs20_20_woRat_noReco_g2d_nobias_nodrop_cu1_PosUp_normed",
    model_checkpoint=None,
):
    torch.cuda.empty_cache()
    model = ProtoTEx_roberta(
        num_prototypes=num_prototypes,
        num_pos_prototypes=num_pos_prototypes,
        class_weights=class_weights,
        bias=False,
        dropout=False,
        special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
        p=1,  # p=0.75,
        batchnormlp1=True,
    ).cuda()

    sampler = StratifiedSampler(
        class_vector=torch.LongTensor(train_dl.dataset.y),
        batch_size=args.num_prototypes,
    )
    random_data_loader = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=args.num_prototypes,
            collate_fn=train_dl.dataset.collate_fn,
            sampler=sampler,
        )

    if model_checkpoint:
        print(f"Loading model checkpoint: {model_checkpoint}")
        pretrained_dict = torch.load(model_checkpoint)
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
    else:
        # TODO: Try getting the average of a first few batches
        batch = next(iter(random_data_loader))
        input_ids, attn_mask, y = batch
        model.set_prototypes(
            input_ids_pos_rdm=input_ids, attn_mask_pos_rdm=attn_mask, do_random=True
        )

    # Track all model parameters
    wandb.watch(
        models=model,
        criterion=model.loss_fn,
        log="all",
        log_freq=len(random_data_loader),
    )

    optim = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, eps=1e-8)

    save_path = MODELPATH + modelname
    logs_path = LOGSPATH + modelname
    f = open(logs_path, "w")
    f.writelines([""])
    f.close()
    val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(val_dl, model)
    epoch = -1
    print_logs(
        logs_path,
        "VAL SCORES",
        epoch,
        val_loss,
        mac_val_prec,
        mac_val_rec,
        mac_val_f1,
        accuracy,
    )
    es = EarlyStopping(-np.inf, patience=10, path=save_path, save_epochwise=False)
    n_iters = 1000
    gamma = 2
    delta = 1
    kappa = 1
    p1_lamb = 0.9
    p2_lamb = 0.9
    p3_lamb = 0.9
    for iter_ in range(n_iters):
        wandb.log({"epoch": iter_})
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
        model.set_protos_status(status=True)
        model.set_classfn_status(status=False)
        #     print("\n after gamma")
        #     for n,w in model.named_parameters():
        #         if w.requires_grad: print(n)
        for epoch in range(delta):
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="delta training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                #             print(y)
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=0,
                    use_rc=0,
                    use_p1=1,
                    use_p2=0,
                    use_p3=0,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    p3_lamb=p3_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                    random_mask_for_distanceMat=None,
                )
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
        model.set_protos_status(status=False)
        model.set_classfn_status(status=True)
        #     print("\n after gamma")
        #     for n,w in model.named_parameters():
        #         if w.requires_grad: print(n)
        for epoch in range(gamma):
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="gamma training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=1,
                    use_rc=0,
                    use_p1=0,
                    use_p2=1,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                )
                optim.zero_grad()
                loss[0].backward()
                optim.step()
        
        
        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            train_dl, model
        )
        print_logs(
            logs_path,
            "TRAIN SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Train epoch": iter_,
                "Train loss": val_loss,
                "Train Precision": mac_val_prec,
                "Train Recall": mac_val_rec, 
                "Train Accuracy": accuracy,
                "Train F1 score": mac_val_f1,
            }
        )
        es.activate(mac_val_f1)
        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            val_dl, model
        )
        print_logs(
            logs_path,
            "VAL SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Val epoch": iter_,
                "Val loss": val_loss,
                "Val Precision": mac_val_prec,
                "Val Recall": mac_val_rec, 
                "Val Accuracy": accuracy,
                "Val F1 score": mac_val_f1,
            }
        )

        es(np.mean(mac_val_f1), epoch, model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "TEST SCORES",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )
            wandb.log(
            {
                "Test epoch": iter_,
                "Test loss": val_loss,
                "Test Precision": mac_val_prec,
                "Test Recall": mac_val_rec, 
                "Test Accuracy": accuracy,
                "Test F1 score": mac_val_f1,
            }
        )

        elif (iter_ + 1) % 5 == 0:

            # print_logs(logs_path,"Early Stopping VAL SCORES (not the best ones)",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1)

            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "Early Stopping TEST SCORES (not the best ones)",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )


def train_ProtoTEx_w_neg_electra(
    train_dl,
    val_dl,
    test_dl,
    num_prototypes,
    num_pos_prototypes,
    class_weights=None,
    modelname="0408_NegProtoBart_protos_xavier_large_bs20_20_woRat_noReco_g2d_nobias_nodrop_cu1_PosUp_normed",
    model_checkpoint=None,
):
    torch.cuda.empty_cache()
    model = ProtoTEx_electra(
        num_prototypes=num_prototypes,
        num_pos_prototypes=num_pos_prototypes,
        class_weights=class_weights,
        bias=False,
        dropout=False,
        special_classfn=True,  # special_classfn=False, ## apply dropout only on bias
        p=1,  # p=0.75,
        batchnormlp1=True,
    ).cuda()

    sampler = StratifiedSampler(
        class_vector=torch.LongTensor(train_dl.dataset.y),
        batch_size=args.num_prototypes,
    )
    random_data_loader = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=args.num_prototypes,
            collate_fn=train_dl.dataset.collate_fn,
            sampler=sampler,
        )

    if model_checkpoint:
        print(f"Loading model checkpoint: {model_checkpoint}")
        pretrained_dict = torch.load(model_checkpoint)
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
    else:
        # TODO: Try getting the average of a first few batches
        batch = next(iter(random_data_loader))
        input_ids, attn_mask, y = batch
        model.set_prototypes(
            input_ids_pos_rdm=input_ids, attn_mask_pos_rdm=attn_mask, do_random=True
        )

    # Track all model parameters
    wandb.watch(
        models=model,
        criterion=model.loss_fn,
        log="all",
        log_freq=len(random_data_loader),
    )

    optim = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, eps=1e-8)

    save_path = MODELPATH + modelname
    logs_path = LOGSPATH + modelname
    f = open(logs_path, "w")
    f.writelines([""])
    f.close()
    val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(val_dl, model)
    epoch = -1
    print_logs(
        logs_path,
        "VAL SCORES",
        epoch,
        val_loss,
        mac_val_prec,
        mac_val_rec,
        mac_val_f1,
        accuracy,
    )
    es = EarlyStopping(-np.inf, patience=10, path=save_path, save_epochwise=False)
    n_iters = 1000
    gamma = 2
    delta = 1
    kappa = 1
    p1_lamb = 0.9
    p2_lamb = 0.9
    p3_lamb = 0.9
    for iter_ in range(n_iters):
        wandb.log({"epoch": iter_})
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
        model.set_protos_status(status=True)
        model.set_classfn_status(status=False)
        #     print("\n after gamma")
        #     for n,w in model.named_parameters():
        #         if w.requires_grad: print(n)
        for epoch in range(delta):
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="delta training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                #             print(y)
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=0,
                    use_rc=0,
                    use_p1=1,
                    use_p2=0,
                    use_p3=0,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    p3_lamb=p3_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                    random_mask_for_distanceMat=None,
                )
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
        model.set_protos_status(status=False)
        model.set_classfn_status(status=True)
        #     print("\n after gamma")
        #     for n,w in model.named_parameters():
        #         if w.requires_grad: print(n)
        for epoch in range(gamma):
            train_loader = tqdm(
                train_dl, total=len(train_dl), unit="batches", desc="gamma training"
            )
            for batch in train_loader:
                input_ids, attn_mask, y = batch
                classfn_out, loss = model(
                    input_ids,
                    attn_mask,
                    y,
                    use_decoder=0,
                    use_classfn=1,
                    use_rc=0,
                    use_p1=0,
                    use_p2=1,
                    rc_loss_lamb=1.0,
                    p1_lamb=p1_lamb,
                    p2_lamb=p2_lamb,
                    distmask_lp1=1,
                    distmask_lp2=1,
                )
                optim.zero_grad()
                loss[0].backward()
                optim.step()
        
        
        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            train_dl, model
        )
        print_logs(
            logs_path,
            "TRAIN SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Train epoch": iter_,
                "Train loss": val_loss,
                "Train Precision": mac_val_prec,
                "Train Recall": mac_val_rec, 
                "Train Accuracy": accuracy,
                "Train F1 score": mac_val_f1,
            }
        )
        es.activate(mac_val_f1)
        val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
            val_dl, model
        )
        print_logs(
            logs_path,
            "VAL SCORES",
            iter_,
            val_loss,
            mac_val_prec,
            mac_val_rec,
            mac_val_f1,
            accuracy,
        )
        wandb.log(
            {
                "Val epoch": iter_,
                "Val loss": val_loss,
                "Val Precision": mac_val_prec,
                "Val Recall": mac_val_rec, 
                "Val Accuracy": accuracy,
                "Val F1 score": mac_val_f1,
            }
        )

        es(np.mean(mac_val_f1), epoch, model)
        if es.early_stop:
            break
        if es.improved:
            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "TEST SCORES",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )
            wandb.log(
                {
                    "Test epoch": iter_,
                    "Test loss": val_loss,
                    "Test Precision": mac_val_prec,
                    "Test Recall": mac_val_rec, 
                    "Test Accuracy": accuracy,
                    "Test F1 score": mac_val_f1,
                }
            )

        elif (iter_ + 1) % 5 == 0:

            # print_logs(logs_path,"Early Stopping VAL SCORES (not the best ones)",iter_,val_loss,mac_val_prec,mac_val_rec,mac_val_f1)

            """
            Below using "val_" prefix but the dl is that of test.
            """
            val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy = evaluate(
                test_dl, model
            )
            print_logs(
                logs_path,
                "Early Stopping TEST SCORES (not the best ones)",
                iter_,
                val_loss,
                mac_val_prec,
                mac_val_rec,
                mac_val_f1,
                accuracy,
            )