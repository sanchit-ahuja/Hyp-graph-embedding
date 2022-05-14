from collections import namedtuple
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import time
import os 
from datetime import datetime
import pickle
import gc
import copy

import torch as th
print(th.cuda.is_available())
import dgl
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from initializer import Initializer
from dataset import create_dataset
from loss import loss_fn

def uniq(labels,num_classes):
    cnt = []
    for i in range(num_classes):
        cnt.append(int(sum(labels==i))+1)
    return cnt


def train_loop(model,data_loader,optimizer,device,h_size,num_classes,beta,gamma):
    train_preds = []
    train_true_l = []
    train_logits = []

    model.train()
    for step, batch in tqdm(enumerate(data_loader),total=len(data_loader)):
        g, emb = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'train')

        true_labels = batch.label[batch.train_mask==1] 

        loss = loss_fn(logits, true_labels, num_classes, uniq(true_labels,num_classes), device, beta, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = th.argmax(logits, 1)

        train_logits.append(logits)
        train_preds.extend(pred.to('cpu'))
        train_true_l.extend(true_labels.to('cpu'))

    train_metrics = model.compute_metrics(train_true_l,train_preds)
    train_logits = th.cat(train_logits).to(device)
    train_true_l = th.tensor(train_true_l).to(device)
    train_loss = loss_fn(train_logits, train_true_l, num_classes, uniq(train_true_l,num_classes), device, beta, gamma)

    print("Train Loss {:.4f} | Train M.F1 {:.4f} | ".format(train_loss.item(),train_metrics['f1']),end='')

    del logits
    del train_metrics
    gc.collect()


def val_loop(model,data_loader,device,h_size,num_classes,beta,gamma):
    val_preds = []
    val_true_l = []
    val_logits = []
    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        with th.no_grad():
            logits = model(batch, h, c, 'val')
        val_logits.append(logits)
        true_labels = batch.label[batch.val_mask==1] 
        pred = th.argmax(logits, 1)
        val_preds.extend(pred.to('cpu'))
        val_true_l.extend(true_labels.to('cpu'))

    val_metrics = model.compute_metrics(val_true_l,val_preds)
    val_logits = th.cat(val_logits).to(device)
    val_true_l = th.tensor(val_true_l).to(device)
    val_loss = loss_fn(val_logits, val_true_l, num_classes, uniq(val_true_l,num_classes), device, beta, gamma)
    val_metrics["loss"] = float(val_loss.item())

    print("Val Loss {:.4f} |".format(val_loss.item()))
    
    gc.collect()

    return val_metrics

def test_loop(model,data_loader,device,h_size,num_classes,beta,gamma,save=False):
    test_preds = []
    test_true_l = []
    test_logits = []
    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        with th.no_grad():
            logits = model(batch, h, c, 'test',save=save)
        test_logits.append(logits)
        true_labels = batch.label[batch.test_mask==1] 
        pred = th.argmax(logits, 1)
        test_preds.extend(pred.to('cpu'))
        test_true_l.extend(true_labels.to('cpu'))

    if save:
        model.save_embs()
        gc.collect()

    test_metrics = model.compute_metrics(test_true_l,test_preds)
    test_logits = th.cat(test_logits).to(device)
    test_true_l = th.tensor(test_true_l).to(device)
    test_loss = loss_fn(test_logits, test_true_l, num_classes, uniq(test_true_l,num_classes), device, beta, gamma)
    test_metrics["loss"] = test_loss.item()

    gc.collect()

    return test_metrics

def main(args):
    start = time.time()

    data_dir = args.data_dir
    x_size = args.x_size
    h_size = args.h_size
    dropout = args.dropout
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    beta = args.beta
    gamma = args.gamma
    batch_size = args.batch_size
    patience = args.patience
    min_epochs = args.min_epochs
    device = args.device
    optim_type = args.optimizer
    save = args.save
    save_dir = args.save_dir
    print(args)

    initializer = Initializer(args)

    print("Loading Dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(data_dir) 
    num_classes = train_dataset.num_classes

    if device=='auto':
        device = 'cuda' if th.cuda.is_available() else 'cpu'
    print("Device:",device)

    model = initializer.initialize_model(num_classes, device)
    model.to(device)
    print(model)

    optimizer = initializer.initialize_optimizer(optim_type)(model.parameters(),lr=lr,weight_decay=weight_decay)
    batcher = initializer.initialize_batcher()

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    return

    counter=0
    best_val_metrics = model.init_metric_dict()
    test_metrics = None
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        print("Epoch: ",epoch)
        train_loop(model,train_loader,optimizer,device,h_size,num_classes,beta,gamma)

        val_metrics = val_loop(model,val_loader,device,h_size,num_classes,beta,gamma)
        if model.has_improved(best_val_metrics, val_metrics):
            test_metrics = test_loop(model,test_loader,device,h_size,num_classes,beta,gamma)
            best_val_metrics = val_metrics
            best_model_wts = copy.deepcopy(model.state_dict())
            counter=0
        else:
            counter += 1
            if counter > patience and epoch > min_epochs:
                print("Early stopping")
                break
    if test_metrics is not None:
        print("(Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(test_metrics["loss"], test_metrics["f1"], test_metrics["recall"]))
        print(test_metrics["conf_mat"])
        mat = test_metrics["conf_mat"]
        print(np.trace(mat)/np.sum(mat))

    if not os.path.exists('results'):
        os.makedirs('results')

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    end = time.time()
    print("Time Elapsed: ",end-start)

    save_dict = {}
    save_dict['test_metrics'] = test_metrics
    save_dict['args'] = args
    save_dict['time'] = current_time

    dir_ = 'results/'+save_dir+'/'+current_time

    model.load_state_dict(best_model_wts)

    if save and test_metrics is not None:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = dir_ + "/" + current_time + ".pkl"

        with open(fname,'wb') as f:
            pickle.dump(save_dict,f)

    return test_metrics
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--x-size', type=int, default=1024)
    parser.add_argument('--h-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('-beta', '--beta', default=0.9999, type=float)
    parser.add_argument('-gamma', '--gamma', default=2.5, type=float)
    parser.add_argument('--data-dir', type=str, default='pheme_dgl_full_roberta_final.pkl',help='directory for data')
    parser.add_argument('--optimizer', type=str, default='Adam',choices=['Adam','RiemannianAdam'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--save-dir', type=str, default='./saved',help='directory for save')
    parser.add_argument('--c', type=float, default=1.0)

    args = parser.parse_args()

    main(args)
