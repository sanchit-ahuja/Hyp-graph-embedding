import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import pickle

from models.base_model import BaseModel
from models.hybrid_manifold_cells import MRIL

from mctorch import nn as mnn
from layers.linear import sLinear

class UniModel(BaseModel):
    def __init__(self, x_size, h_size, num_classes, dropout, device, c=1.0):
        super().__init__()

        self.weight_manifold = mnn.Stiefel
        self.p_manifold = "custom"

        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.root_linear = sLinear(x_size,h_size//2, weight_manifold=self.weight_manifold, projective_manifold=None, c=c)
        self.linear1 = sLinear(h_size, h_size//4, weight_manifold=self.weight_manifold, projective_manifold=None, c=c)

        self.cell = MRIL(x_size, h_size, device, c)

        self.device = device

        self.save_data = []

    def forward(self, batch, h, c, mode='train',save=False):
        g = batch.graph.to(self.device)
        embeds = batch.feats.to(self.device)

        g.ndata['iou1'] = self.cell.W_iou(self.cell.pmath_geo1.expmap0(self.dropout(embeds)), pmath_geo=self.cell.pmath_geo1)
        g.ndata['iou2'] = self.cell.W_iou(self.cell.pmath_geo2.expmap0(self.dropout(embeds)), pmath_geo=self.cell.pmath_geo2)
        g.ndata['iou3'] = self.cell.W_iou(self.dropout(embeds))

        g.ndata['h1'] = h
        g.ndata['c1'] = c
        g.ndata['h2'] = h
        g.ndata['c2'] = c
        g.ndata['h3'] = h
        g.ndata['c3'] = c

        dgl.prop_nodes_topo(g, message_func=self.cell.message_func, reduce_func=self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)

        h1 = self.dropout(self.cell.pmath_geo1.logmap0(g.ndata.pop('h1')))
        h2 = self.dropout(self.cell.pmath_geo2.logmap0(g.ndata.pop('h2')))
        h3 = self.dropout(g.ndata.pop('h3'))
        h = torch.stack([h1,h2,h3],dim=1)
        
        h = self.linear1(h)
        if mode=='train':
            mask = batch.train_mask
        elif mode=='val':
            mask = batch.val_mask
        elif mode == 'test':
            mask = batch.test_mask

        if save:
            g.ndata['embs'] = h

        h = h[mask==1]
        logits = h

        if save:
            self.save_data.append((g.cpu(),logits,g.ndata['y']))

        return nn.functional.gelu(logits)
        
    def save_embs(self):
        with open("embeddings.pkl",'wb') as f:
            pickle.dump(self.save_data,f)