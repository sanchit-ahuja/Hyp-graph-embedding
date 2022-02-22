import dgl
import pickle
import numpy as np
import os
import torch
import networkx as nx
from collections import namedtuple
import random
import signal
from tqdm import tqdm

class Processor():
    def __init__(self, data_dir):

        self.attrs = ['x', 'y', 'del_t', 'train_mask', 'val_mask', 'test_mask']
        self.num_classes = 2
        filepath = os.path.join(data_dir,"pheme_graph.pkl")

        with open(filepath,"rb") as f:
            dataset = pickle.load(f)

        self.trees = dataset
        self.labels = None

    def __getitem__(self,item):
        return self.trees[item]
    
    def __len__(self):
        return len(self.trees)
    
    def get_labels(self):
        return self.labels


class DAGDataset(torch.utils.data.Dataset):
    def __init__(self, trees, labels, num_classes=2):
        
        self.trees = trees
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self,item):
        return self.trees[item]
    
    def __len__(self):
        return len(self.trees)
    
    def get_labels(self):
        return self.labels


def create_dataset(data_dir):
    processor = Processor(data_dir)
    num_classes = processor.num_classes
    train_trees = []
    train_labels = []
    val_trees = []
    val_labels = []
    test_trees = []
    test_labels = []

    processor = process_split(processor)

    if processor.labels:
        for t, l in zip(processor.trees, processor.labels):
            if(sum(t.ndata['train_mask'])>=1):
                train_trees.append(t)
                train_labels.append(l)
            if(sum(t.ndata['val_mask'])>=1):
                val_trees.append(t)
                val_labels.append(l)
            if(sum(t.ndata['test_mask'])>=1):
                test_trees.append(t)
                test_labels.append(l)
    else:
        for t in processor.trees:
            if(sum(t.ndata['train_mask'])>=1):
                train_trees.append(t)
            if(sum(t.ndata['val_mask'])>=1):
                val_trees.append(t)
            if(sum(t.ndata['test_mask'])>=1):
                test_trees.append(t)
    print(len(train_trees),len(val_trees),len(test_trees))
        

    return DAGDataset(train_trees,train_labels,num_classes), DAGDataset(val_trees,val_labels,num_classes), DAGDataset(test_trees,test_labels,num_classes)



def process_split(processor):
    attrs = ['x','y','train_mask','val_mask','test_mask','del_t']
    trees = [dgl.to_networkx(t,node_attrs=attrs) for t in processor.trees]
    new_trees = trees

    random.shuffle(new_trees)
    l_size = len(new_trees)
    k = 0
    final_trees = []
    for tree in new_trees:
        for node in tree:
            if tree.out_degree(node)==0:
                if k < 0.7 * l_size:
                    tree.nodes[node]['train_mask'] = 1
                    tree.nodes[node]['val_mask'] = 0
                    tree.nodes[node]['test_mask'] = 0 
                elif k < 0.8 *l_size:   
                    tree.nodes[node]['train_mask'] = 0
                    tree.nodes[node]['val_mask'] = 1
                    tree.nodes[node]['test_mask'] = 0
                else:
                    tree.nodes[node]['train_mask'] = 0
                    tree.nodes[node]['val_mask'] = 0
                    tree.nodes[node]['test_mask'] = 1      
                k+=1
                break
        final_trees.append(tree)

    new_trees = [dgl.from_networkx(t,node_attrs=attrs) for t in final_trees] 
    processor.trees = new_trees
    return processor