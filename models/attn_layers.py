import torch
import torch.nn as nn
import numpy as np
import math
from geoopt.manifolds.stereographic.manifold import PoincareBall


perterb = 1e-15

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    def single_query_attn_scores(self, key, query):
        euclid_key = key
        euclid_query = query
        scores = torch.bmm(euclid_key, euclid_query.unsqueeze(-1))
        denom = torch.norm(euclid_key)
        scores = (1./ denom) * scores
        return scores

    def forward(self, key, query, value, seq_lens=None):

        scores = self.single_query_attn_scores(key, query)
        scaled_scores = torch.nn.functional.softmax(scores, -2)
        if seq_lens is not None:
            mask = torch.ones_like(scaled_scores).squeeze().type(
                value.dtype).detach()
            for id_in_batch, seq_len in enumerate(seq_lens):
                mask[id_in_batch, seq_len:] = 0.
            scaled_scores = scaled_scores.squeeze() * mask
            _sums = scaled_scores.sum(-1, keepdim=True)  # sums per row
            scaled_scores = scaled_scores.div(_sums).unsqueeze(-1)
        scaled_scores = scaled_scores + perterb
        out = scaled_scores.shape[1] * torch.sum(value*scaled_scores,1)
        return out


class HyperAttn(nn.Module):
    def __init__(self, manifold):
        super(HyperAttn, self).__init__()
        self.beta = nn.Parameter(torch.Tensor([1.0]),requires_grad=True)
        self.c = nn.Parameter(torch.Tensor([0.0]),requires_grad=True)
        self.manifold = manifold

    def single_query_attn_scores(self, key, query):
        scores = self.manifold.dist(key,query.unsqueeze(dim=1))
        return self.beta*scores - self.c


    def forward(self, key, query, value, reducedim=[1]):
        scores = self.single_query_attn_scores(key, query) 
        scaled_scores = torch.nn.functional.softmax(scores, -2)  
        scaled_scores = scaled_scores + perterb
        out = scaled_scores.shape[1] * self.manifold.weighted_midpoint(xs=value, weights=scaled_scores, reducedim=reducedim)
        return out