import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import math
import itertools

from geoopt.manifolds.stereographic.manifold import PoincareBall, SphereProjection
from mctorch import nn as mnn

from models.attn_layers import HyperAttn, Attention
from layers.linear import sLinear


class MRIL(torch.nn.Module):
    def __init__(self, x_size, h_size, device, c=1.0):
        super(MRIL, self).__init__()

        projective_manifold = "custom"
        weight_manifold =mnn.Stiefel

        self.W_iou = sLinear(x_size, 3 * h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.U_iou = sLinear(h_size, 3 * h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.W_f = sLinear(x_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.U_f = sLinear(h_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.W_q = sLinear(x_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.W_k = sLinear(h_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.W_c = sLinear(h_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.U_s = sLinear(h_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        self.U_p = sLinear(h_size, h_size, weight_manifold=weight_manifold, projective_manifold=projective_manifold, c=c)
        
        self.temp = torch.tensor([1e-6]).to(device)
        self.const_bias_param = torch.nn.Parameter(torch.Tensor(5))
        self.device = device
        self.x_size = x_size
        self.h_size = h_size

        self.pmath_geo1 = PoincareBall(c=c)
        self.pmath_geo2 = SphereProjection(c)
        self.attn1 = HyperAttn(self.pmath_geo1)
        self.attn2 = HyperAttn(self.pmath_geo2)
        self.attn3 = Attention()
        self.b = nn.Parameter(torch.Tensor([1e-6]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([0.02]), requires_grad=True)
        self.d = nn.Parameter(torch.Tensor([60.0]), requires_grad=True)


    def message_func(self, edges):
        return {'h1': edges.src['h1'], 'c1': edges.src['c1'], 'h2': edges.src['h2'], 'c2': edges.src['c2'],  'h3': edges.src['h3'], 'c3': edges.src['c3'], 'del_t': edges.src['del_t']}

    def reduce_func(self, nodes):

        del_t = nodes.mailbox['del_t']
        g_t = self.d/(del_t + 1)

        x_1p = self.W_q(self.pmath_geo1.expmap0(nodes.data['x']), pmath_geo=self.pmath_geo1)
        x_1s = self.W_q(self.pmath_geo2.expmap0(nodes.data['x']), pmath_geo=self.pmath_geo2)
        x_1e = self.W_q(nodes.data['x'])

        h1 = nodes.mailbox['h1']
        h2 = nodes.mailbox['h2']
        h3 = nodes.mailbox['h3']


        s1 = self.pmath_geo1.logmap0(torch.sigmoid(self.U_p(h1, pmath_geo=self.pmath_geo1)))
        s2 = self.pmath_geo2.logmap0(torch.sigmoid(self.U_p(h2, pmath_geo=self.pmath_geo2)))
        s3 = torch.sigmoid(self.U_p(h3))
        
        h_tild_1p = self.pmath_geo1.mobius_pointwise_mul(s1, h1)
        h_tild_2p = self.pmath_geo1.mobius_pointwise_mul(s2, self.pmath_geo1.expmap0(self.pmath_geo2.logmap0(h2)))
        h_tild_3p = self.pmath_geo1.mobius_pointwise_mul(s3, self.pmath_geo1.expmap0(h3))

        h_1p = 3 * self.pmath_geo1.weighted_midpoint(xs=torch.stack([h_tild_1p,h_tild_2p,h_tild_3p]),reducedim=[0])

        
        h_tild_1s = self.pmath_geo1.mobius_pointwise_mul(s1, self.pmath_geo2.expmap0(self.pmath_geo1.logmap0(h1)))
        h_tild_2s = self.pmath_geo2.mobius_pointwise_mul(s2, h2)
        h_tild_3s = self.pmath_geo2.mobius_pointwise_mul(s3, self.pmath_geo2.expmap0(h3))
        h_1s = 3 * self.pmath_geo1.weighted_midpoint(xs=torch.stack([h_tild_1s,h_tild_2s,h_tild_3s]),reducedim=[0])
        
        h_tild_1e = self.pmath_geo1.logmap0(self.pmath_geo1.mobius_pointwise_mul(s1, h1))
        h_tild_2e = self.pmath_geo2.logmap0(self.pmath_geo2.mobius_pointwise_mul(s2, h2))
        h_tild_3e = s3 * h3
        h_1e = torch.sum(torch.stack([h_tild_1e,h_tild_2e,h_tild_3e]), dim=0)

        h_tildp = self.attn1(h_1p, x_1p, h_1p, reducedim=[1])
        h_tilds = self.attn2(h_1s, x_1s, h_1s, reducedim=[1])
        h_tilde = self.attn3(h_1e, x_1e, h_1e)

        c_k_p = nodes.mailbox['c1']
        c_sk_p = self.pmath_geo1.expmap0(torch.tanh(self.pmath_geo1.logmap0(self.W_c(c_k_p, pmath_geo=self.pmath_geo1))))
        c_sk_hat_p = self.pmath_geo1.mobius_pointwise_mul(c_sk_p, g_t.unsqueeze(dim=-1))
        c_Tk_p = self.pmath_geo1.mobius_add(-c_sk_p, c_k_p)
        c_k_tilde_p = self.pmath_geo1.mobius_add(c_Tk_p, c_sk_hat_p)
        f_p = torch.sigmoid(self.pmath_geo1.logmap0(self.U_f(h1, pmath_geo=self.pmath_geo1)))
        c_1 = self.pmath_geo1.weighted_midpoint(self.pmath_geo1.mobius_pointwise_mul(f_p, c_k_tilde_p), reducedim=[1])

        c_k_s = nodes.mailbox['c2']
        c_sk_s = self.pmath_geo2.expmap0(torch.tanh(self.pmath_geo2.logmap0(self.W_c(c_k_s, pmath_geo=self.pmath_geo2))))
        c_sk_hat_s = self.pmath_geo2.mobius_pointwise_mul(c_sk_s, g_t.unsqueeze(dim=-1))
        c_Tk_s = self.pmath_geo2.mobius_add(-c_sk_s, c_k_s)
        c_k_tilde_s = self.pmath_geo2.mobius_add(c_Tk_s, c_sk_hat_s)
        f_s = torch.sigmoid(self.pmath_geo2.logmap0(self.U_f(h2, pmath_geo=self.pmath_geo2)))
        c_2 = self.pmath_geo2.weighted_midpoint(self.pmath_geo2.mobius_pointwise_mul(f_s, c_k_tilde_s), reducedim=[1])

        c_k_e = nodes.mailbox['c3']
        c_sk_e = torch.tanh(self.W_c(c_k_e))
        c_sk_hat_e = c_sk_e * g_t.unsqueeze(dim=-1)
        c_Tk_e = c_k_e - c_sk_e
        c_k_tilde_e = c_Tk_e + c_sk_hat_e  
        f_e = torch.sigmoid(self.U_f(h3))
        c_3 = torch.sum(f_e * c_k_tilde_e, 1)

        iou1 = self.pmath_geo1.mobius_add(nodes.data['iou1'], self.U_iou(h_tildp, pmath_geo=self.pmath_geo1))
        iou2 = self.pmath_geo2.mobius_add(nodes.data['iou2'], self.U_iou(h_tilds, self.pmath_geo2))
        iou3 = nodes.data['iou3'] + self.U_iou(h_tilde)

        return {'iou1': iou1, 'iou2': iou2, 'iou3': iou3, 'c1': c_1, 'c2': c_2, 'c3': c_3}

    def apply_node_func(self, nodes):
        iou_p = nodes.data['iou1']
        i_p, o_p, u_p = torch.chunk(iou_p, 3, 1)
        i_p, o_p, u_p = torch.sigmoid(self.pmath_geo1.logmap0(i_p)), torch.sigmoid(self.pmath_geo1.logmap0(o_p)), torch.tanh(self.pmath_geo1.logmap0(u_p))

        iou_s = nodes.data['iou2']
        i_s, o_s, u_s = torch.chunk(iou_s, 3, 1)
        i_s, o_s, u_s = torch.sigmoid(self.pmath_geo2.logmap0(i_s)), torch.sigmoid(self.pmath_geo2.logmap0(o_s)), torch.tanh(self.pmath_geo2.logmap0(u_s))

        iou_e = nodes.data['iou3']
        i_e, o_e, u_e = torch.chunk(iou_e, 3, 1)
        i_e, o_e, u_e = torch.sigmoid(i_e), torch.sigmoid(o_e), torch.tanh(u_e)

        c1 = self.pmath_geo1.mobius_add(self.pmath_geo1.mobius_pointwise_mul(i_p, u_p), nodes.data['c1'])
        c2 = self.pmath_geo2.mobius_add(self.pmath_geo2.mobius_pointwise_mul(i_s, u_s), nodes.data['c2'])
        c3 = i_e * u_e + nodes.data['c3']
        h1 = self.pmath_geo1.mobius_pointwise_mul(o_p, torch.tanh(self.pmath_geo1.logmap0(c1)))
        h2 = self.pmath_geo2.mobius_pointwise_mul(o_s, torch.tanh(self.pmath_geo2.logmap0(c2)))
        h3 = o_e * torch.tanh(c3)
        
        return {'h1': h1, 'c1': c1, 'h2': h2, 'c2': c2, 'h3': h3, 'c3': c3}