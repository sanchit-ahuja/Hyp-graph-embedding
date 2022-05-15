import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import numpy as np
import gc
from loss import loss_fn
from sklearn.metrics import f1_score
from dataset import create_dataset

# from loss import loss_fn
from vanilla_model import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys

logging.basicConfig(
    filename="logs/vanilla_model.log",
    level=logging.INFO,
    filemode="w",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def uniq(labels, num_classes):
    cnt = []
    for i in range(num_classes):
        cnt.append(int(sum(labels == i)) + 1)
    return cnt


def train(train_loader, model, loss_fn, optim, num_classes, device, beta, gamma):
    train_preds = []
    train_true_l = []
    train_logits = []
    model.train()
    loss_total = 0
    for data in train_loader:
        x_feats = data["x"].to(device)
        labels = data["y"].to(device)
        # del_t = data.ndata['del_t']
        train_mask = data["train_mask"].to(device)
        true_labels = labels[train_mask == 1]
        logits = model(x_feats)
        loss = loss_fn(
            logits,
            true_labels,
            num_classes,
            uniq(true_labels, num_classes),
            device,
            beta,
            gamma,
        )
        # loss = loss_fn(logits, true_labels)
        loss_total += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = logits.argmax(1)
        train_logits.extend(logits)
        train_preds.extend(pred)
        train_true_l.extend(true_labels)

    train_logits = torch.cat(train_logits).to(device)
    # convert train_logits to numpy
    train_true_l = torch.tensor(train_true_l).to(device)
    train_true_l = train_true_l.detach().cpu().numpy()
    train_preds = torch.tensor(train_preds).to(device)
    train_preds = train_preds.detach().cpu().numpy()
    train_loss = loss_total / len(train_loader)
    # train_loss = loss_fn(train_logits, train_truoe_l)
    f1_train = f1_score(train_true_l, train_preds, average="macro")
    logger.info("-----Train-----")
    logger.info(f"loss: {train_loss}, f1: {f1_train}")
    del logits
    gc.collect()


def val(val_loader, model, loss_fn, num_classes, device, beta, gamma):
    val_preds = []
    val_true_l = []
    val_logits = []
    val_metrics = {}
    loss_total = 0
    model.eval()
    for data in val_loader:
        x_feats = data["x"].to(device)
        labels = data["y"].to(device)
        val_mask = data["val_mask"].to(device)
        true_labels = labels[val_mask == 1]
        logits = model(x_feats)
        # loss = loss_fn(logits, true_labels)
        loss = loss_fn(
            logits,
            true_labels,
            num_classes,
            uniq(true_labels, num_classes),
            device,
            beta,
            gamma,
        )
        loss_total += loss.item()
        with torch.no_grad():
            logits = model(x_feats)
        val_logits.append(logits)
        pred = torch.argmax(logits, 1)
        val_preds.extend(pred.to("cpu"))
        val_true_l.extend(true_labels.to("cpu"))
    val_logits = torch.cat(val_logits).to(device)
    val_true_l = torch.tensor(val_true_l).to(device)
    val_true_l = val_true_l.detach().cpu().numpy()
    val_preds = torch.tensor(val_preds).to(device)
    val_preds = val_preds.detach().cpu().numpy()
    val_metrics["val_loss"] = loss_total / len(val_loader)
    val_metrics["val_f1"] = f1_score(val_true_l, val_preds, average="macro")
    logger.info("-----Validation-----")
    logger.info(f"val_loss: {val_metrics['val_loss']}, val_f1: {val_metrics['val_f1']}")
    return val_metrics


def test(model, data_loader, device, num_classes, beta, gamma, loss_fn, save=False):
    test_preds = []
    test_true_l = []
    test_logits = []
    model.eval()
    for data, hyp_embeddings in data_loader:
        x_feats = data.ndata["x"]
        labels = data.data["y"]
        test_mask = data.ndata["test_mask"]
        with torch.no_grad():
            logits = model(x_feats, hyp_embeddings)
            # logits = model(batch, h, c, 'test',save=save)
        test_logits.append(logits)
        true_labels = labels[test_mask == 1]
        pred = torch.argmax(logits, 1)
        test_preds.extend(pred.to("cpu"))
        test_true_l.extend(true_labels.to("cpu"))

    if save:
        model.save_embs()
        gc.collect()

    test_metrics = model.compute_metrics(test_true_l, test_preds)
    test_logits = torch.cat(test_logits).to(device)
    test_true_l = torch.tensor(test_true_l).to(device)
    test_loss = loss_fn(
        test_logits,
        test_true_l,
    )
    test_metrics["test_loss"] = test_loss.item()

    gc.collect()

    return test_metrics


def main(args):
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
    print("Loading Dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(data_dir)
    num_classes = train_dataset.num_classes

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model = BaseModel(num_classes=num_classes, hidden_size=h_size, dropout=dropout)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # return
    # loss_fn = nn.CrossEntropyLoss()
    counter = 0
    val_loss_min = np.inf
    for epoch in tqdm(range(epochs)):
        print("Epoch:", epoch)
        train(train_loader, model, loss_fn, optim, num_classes, device, beta, gamma)
        val_metrics = val(val_loader, model, loss_fn, num_classes, device, beta, gamma)
        if val_metrics["val_loss"] < val_loss_min:
            counter = 0
            val_loss_min = val_metrics["val_loss"]
            print("SAVING")
            torch.save(
                model.state_dict(), save_dir + f"/{val_metrics['val_f1']}_model.pt"
            )
        else:
            counter += 1
            if counter == patience:
                print("Early Stopping")
                break

    pass


def has_improved(m1, m2):
    return m1["val_loss"] > m2["val_loss"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--x-size", type=int, default=1024)
    parser.add_argument("--h-size", type=int, default=1026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("-beta", "--beta", default=0.9999, type=float)
    parser.add_argument("-gamma", "--gamma", default=2.5, type=float)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="pheme_dgl_full_roberta_final.pkl",
        help="directory for data",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam", "RiemannianAdam"]
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--save_dir", type=str, default="./saved", help="directory for save"
    )
    parser.add_argument("--c", type=float, default=1.0)

    args = parser.parse_args()
    logger.info("-----PARAMS-----")
    logger.info(args)
    main(args)
