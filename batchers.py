import dgl
import pickle
import numpy as np
import os
import torch
print(torch.cuda.is_available())
import networkx as nx
from collections import namedtuple


def dag_batcher(device):
    def batcher_dev(batch):
        TreeBatch = namedtuple('DAGBatch', ['graph', 'feats', 'label', 'train_mask', 'val_mask', 'test_mask','del_t'])
        batch_trees = dgl.batch(batch)
        return TreeBatch(graph=batch_trees,train_mask=batch_trees.ndata["train_mask"].to(device),val_mask=batch_trees.ndata["val_mask"].to(device),
            test_mask=batch_trees.ndata["test_mask"].to(device),feats=batch_trees.ndata['x'].to(device),label=batch_trees.ndata['y'].to(device),
            del_t=batch_trees.ndata['del_t'])
    return batcher_dev