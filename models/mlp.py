import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import pickle

from models.uni_model import UniModel
from models.base_model import BaseModel

from mctorch import nn as mnn
from layers.linear import sLinear

class MLP(BaseModel):
    def __init__(self, x_size, h_size, num_classes, dropout, device, time_aware=False, matrix_manifold='stiefel', projective_manifold='custom', c=1.0):
        super().__init__()

        self.device = device
        self.projective_manifold = projective_manifold

        weight_manifold = mnn.Stiefel
        m_manif = weight_manifold

        self.model = UniModel(x_size,h_size,num_classes,dropout,device,c).to(device)
        self.linear2 = sLinear(h_size//4, num_classes, weight_manifold=weight_manifold, projective_manifold=None, c=0)

    def forward(self, batch, h, c, mode='train',save=False):
        outputs = self.model(batch,h,c,mode,save)
        if self.projective_manifold == 'custom':
            outputs, _ = torch.max(outputs, dim=-2, keepdim=False)
        outputs = nn.functional.relu(self.linear2(outputs))
        return outputs
    
    def save_embs(self):
        self.model.save_embs()
