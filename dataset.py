import dgl
import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F
import networkx as nx
from collections import namedtuple
import random
import signal
from batchers import dag_batcher
from torch.utils.data import DataLoader
from tqdm import tqdm


class Processor:
    def __init__(self, filepath):

        self.attrs = ["x", "y", "del_t", "train_mask", "val_mask", "test_mask"]
        self.num_classes = 2
        # filepath = os.path.join(data_dir,"pheme_graph.pkl")

        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
            graphs = dataset["ds"]
            fin_embeddings = dataset["hyp_embedding"]

        self.trees = graphs
        self.labels = None
        self.fin_embeddings = fin_embeddings

    def __getitem__(self, item):
        return self.trees[item], self.fin_embeddings[item]

    def __len__(self):
        return len(self.trees)

    def get_labels(self):
        return self.labels


class DAGDataset(torch.utils.data.Dataset):
    def __init__(self, trees, fin_embeddings, labels, num_classes=2):
        # def iterate_tree(trees, type_mask):
        #     x_feats = []
        #     y_feats = []
        #     masks = []
        #     for ds in trees:
        #         x_feats.append(ds.ndata['x'])
        #         y_feats.append(ds.ndata['y'])
        #         if type_mask == 'train':
        #             masks.append(ds.ndata['train_mask'])
        #         elif type_mask == 'val':
        #             masks.append(ds.ndata['val_mask'])
        #         elif type_mask == 'test':
        #             masks.append(ds.ndata['test_mask'])
        #     return x_feats, y_feats, masks

        # if type_mask not in ['train', 'val', 'test']:
        #     raise ValueError('type_mask must be one of "train", "val", "test"')
        # self.x_feats, self.y_feats, self.masks = iterate_tree(trees, type_mask)
        self.trees = trees
        self.pad_len = max([len(tree) for tree in trees])
        self.labels = labels
        self.num_classes = num_classes
        self.fin_embeddings = fin_embeddings

    def __getitem__(self, item):
        # pad train_mask till max_len by adding 0
        train_mask = self.trees[item].ndata["train_mask"].tolist()
        train_mask = train_mask + [0] * (self.pad_len - len(train_mask))
        val_mask = self.trees[item].ndata["val_mask"].tolist()
        val_mask = val_mask + [0] * (self.pad_len - len(val_mask))
        test_mask = self.trees[item].ndata["test_mask"].tolist()
        test_mask = test_mask + [0] * (self.pad_len - len(test_mask))
        y = self.trees[item].ndata["y"].tolist()
        y = y + [0] * (self.pad_len - len(y))

        return {
            "x": self.fin_embeddings[item].to(torch.float32),
            "y": torch.Tensor(y).to(torch.long),
            "train_mask": torch.Tensor(train_mask).to(torch.long),
            "val_mask": torch.Tensor(val_mask).to(torch.long),
            "test_mask": torch.Tensor(test_mask).to(torch.long),
        }

    def __len__(self):
        return len(self.trees)

    def get_labels(self):
        return self.labels


def create_dataset(data_dir):
    processor = Processor(data_dir)
    num_classes = processor.num_classes
    train_trees = []
    train_fin_embeddings = []
    train_labels = []
    val_trees = []
    val_fin_embeddings = []
    val_labels = []
    test_trees = []
    test_fin_embeddings = []
    test_labels = []

    processor = process_split(processor)

    if processor.labels:
        for t, l, h in zip(processor.trees, processor.labels, processor.fin_embeddings):
            if sum(t.ndata["train_mask"]) >= 1:
                # get index where train_mask is 1
                train_idx = torch.where(t.ndata["train_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][train_idx].to("cuda")
                hyp_feat = h[train_idx].to("cuda")
                train_trees.append(t)
                train_labels.append(l)
                fin_feat = torch.cat([x_feat, hyp_feat], dim=0)
                train_fin_embeddings.append(fin_feat)
            if sum(t.ndata["val_mask"]) >= 1:
                val_idx = torch.where(t.ndata["val_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][val_idx].to("cuda")
                hyp_feat = h[val_idx].to("cuda")
                val_trees.append(t)
                val_labels.append(l)
                fin_feat = torch.cat([x_feat, hyp_feat], dim=0)
                val_fin_embeddings.append(fin_feat)
            if sum(t.ndata["test_mask"]) >= 1:
                test_idx = torch.where(t.ndata["test_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][test_idx].to("cuda")
                hyp_feat = h[test_idx].to("cuda")
                test_trees.append(t)
                test_labels.append(l)
                fin_feat = torch.cat([x_feat, hyp_feat], dim=0)
                test_fin_embeddings.append(fin_feat)
    else:
        for t, h in zip(processor.trees, processor.fin_embeddings):
            if sum(t.ndata["train_mask"]) >= 1:
                train_idx = torch.where(t.ndata["train_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][train_idx].to("cuda")
                hyp_feat = h[train_idx].to("cuda")
                fin_feat = torch.cat([x_feat, hyp_feat], dim=0)
                train_trees.append(t)
                train_fin_embeddings.append(fin_feat)
            if sum(t.ndata["val_mask"]) >= 1:
                val_idx = torch.where(t.ndata["val_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][val_idx].to("cuda")
                hyp_feat = h[val_idx].to("cuda")
                fin_feat = torch.cat([x_feat, hyp_feat], dim=0)
                val_trees.append(t)
                val_fin_embeddings.append(fin_feat)
            if sum(t.ndata["test_mask"]) >= 1:
                test_idx = torch.where(t.ndata["test_mask"] == 1)[0][0].item()
                x_feat = t.ndata["x"][test_idx].to("cuda")
                hyp_feat = h[test_idx].to("cuda")
                test_trees.append(t)
                test_fin_embeddings.append(torch.cat([x_feat, hyp_feat], dim=0))
    print(len(train_trees), len(val_trees), len(test_trees))

    return (
        DAGDataset(train_trees, train_fin_embeddings, train_labels, num_classes),
        DAGDataset(val_trees, val_fin_embeddings, val_labels, num_classes),
        DAGDataset(test_trees, test_fin_embeddings, test_labels, num_classes),
    )


def process_split(processor):
    attrs = ["x", "y", "train_mask", "val_mask", "test_mask", "del_t"]
    trees = [dgl.to_networkx(t, node_attrs=attrs) for t in processor.trees]
    new_trees = trees

    random.shuffle(new_trees)
    l_size = len(new_trees)
    k = 0
    final_trees = []
    for tree in new_trees:
        for node in tree:
            if tree.out_degree(node) == 0:
                if k < 0.7 * l_size:
                    tree.nodes[node]["train_mask"] = 1
                    tree.nodes[node]["val_mask"] = 0
                    tree.nodes[node]["test_mask"] = 0
                elif k < 0.8 * l_size:
                    tree.nodes[node]["train_mask"] = 0
                    tree.nodes[node]["val_mask"] = 1
                    tree.nodes[node]["test_mask"] = 0
                else:
                    tree.nodes[node]["train_mask"] = 0
                    tree.nodes[node]["val_mask"] = 0
                    tree.nodes[node]["test_mask"] = 1
                k += 1
                break
        final_trees.append(tree)

    new_trees = [dgl.from_networkx(t, node_attrs=attrs) for t in final_trees]
    processor.trees = new_trees
    return processor


if __name__ == "__main__":
    data_dir = "pheme_tmp.pkl"
    train_dataset, val, test = create_dataset(data_dir)
    device = "cuda"

    num_classes = train_dataset.num_classes
    batch_size = 8
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    gg = next(iter(train_loader))
    print(gg['x'].shape,'x')
    print(gg['y'].shape,'y')
    # print(gg[0], "c")
    # print(gg[1].shape, 'ch')
    # print(len(gg), "len")
    # print(gg,"check")
    # model = initializer.initialize_model(num_classes, device)
    # model.to(device)
    # print(model)

    # optimizer = initializer.initialize_optimizer(optim_type)(model.parameters(),lr=lr,weight_decay=weight_decay)
    # batcher = initializer.initialize_batcher()
    # batch_size = 128
    # train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    # print(train_dataset[0])
    # print(len(train),len(val),len(test))
