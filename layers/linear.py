import math
import itertools

import torch
from torch.nn import Linear, init
from torch.nn import functional as F

from mctorch.nn.parameter import Parameter
from mctorch.nn.manifolds import create_manifold_parameter, manifold_random_
from mctorch import nn as mnn
from geoopt.manifolds.stereographic.manifold import PoincareBall, SphereProjection

class sLinear(torch.nn.Linear):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True,
                 weight_manifold=None, projective_manifold=None, transpose_flag=False, c=1.0):
        super(sLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight_manifold = weight_manifold
        self.transpose_flag = transpose_flag
        self.weight_transform = lambda x : x

        self.projective_manifold = projective_manifold
        if self.projective_manifold == 'custom':
            self.pmath_geo = 'custom'
        else:
            self.pmath_geo = None

        if weight_manifold is None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.transpose_flag, self.weight = create_manifold_parameter(
                weight_manifold, (out_features, in_features), transpose_flag, self.pmath_geo)
            if self.transpose_flag:
                self.weight_transform = lambda x : x.transpose(-2, -1)

        self.local_reset_parameters()

    def local_reset_parameters(self):
        if self.weight_manifold is not None:
            manifold_random_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_transform(self.weight))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_, pmath_geo=None):
        if pmath_geo is None or self.pmath_geo is None:
            return F.linear(input_, self.weight_transform(self.weight), self.bias)

        weights = self.weight_transform(self.weight)

        if self.pmath_geo =='custom':
            return pmath_geo.mobius_add(pmath_geo.mobius_matvec(weights,input_),self.bias)

        return self.pmath_geo.mobius_add(self.pmath_geo.mobius_matvec(weights,input_),self.bias)

