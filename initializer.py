from models.uni_model import UniModel
from models.mlp import MLP
import torch
from geoopt.optim.radam import RiemannianAdam
from batchers import dag_batcher

class Initializer():
    def __init__(self,args):
        self.data_dir = args.data_dir
        self.x_size = args.x_size
        self.h_size = args.h_size
        self.dropout = args.dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.beta = args.beta
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.patience = args.patience
        self.min_epochs = args.min_epochs
        self.c = args.c

    def initialize_model(self,num_classes, device):
        model = MLP(self.x_size,self.h_size,num_classes,self.dropout,device,self.c)
        return model

    def initialize_optimizer(self,type='Adam'):
        if type == 'Adam':
            return torch.optim.Adam
        elif type == 'RiemannianAdam':
            return RiemannianAdam
        return None

    def initialize_batcher(self):
        return dag_batcher


