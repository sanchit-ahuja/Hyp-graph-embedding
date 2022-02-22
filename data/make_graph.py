import csv
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import pickle
import networkx as nx
import copy
import torch
import dgl

path = "./"

with open(os.path.join(path,'./pheme_roberta_1024.pkl'),'rb') as f:
    df = pickle.load(f)
df=df.astype({'tweet_id': str })
df=df.astype({'parent_id': str })

with open(os.path.join(path,'./df_pheme_for_enc.pkl'),'rb') as f:
    df1 = pickle.load(f)
df1=df1.astype({'tweet_id': str })
df1=df1.astype({'parent_id': str })

df_new = pd.merge(df[['tweet_id','enc']],df1,how='inner',on=['tweet_id'])
df_new = df_new.drop_duplicates('tweet_id')

def process_attrs(G,row,count):
    node = row['tweet_id']
    label = row['label']
    G.nodes[node]['y'] = label 
    if row['enc'].shape[0]==1:
        G.nodes[node]['x'] = np.zeros((1024,))
    else:
        G.nodes[node]['x'] = row['enc']
    G.nodes[node]['tweet_id'] = node
    #try:
    date = datetime.strptime(row["date"], '%a %b %d %H:%M:%S %z %Y')
    G.nodes[node]['timestamp'] = datetime.timestamp(date)
    return G, count

G = nx.DiGraph()


count=0
for idx, row in tqdm(df.iterrows(),total=len(df)):
    tweet_id = str(row["tweet_id"])
    G.add_node(tweet_id)
    parent_id = str(row["parent_id"])
    if parent_id!='None':
        G.add_edge(tweet_id,parent_id)
    G,count = process_attrs(G,row,count)

for node in G:
    edges = G.out_edges(node)
    if(len(edges)==0):
        G.nodes[node]['del_t'] = 0.0
        continue
    parent = list(edges)[0][1]
    try:
        diff = G.nodes[node]['timestamp'] - G.nodes[parent]['timestamp']
    except:
        diff = 60.0
    G.nodes[node]['del_t'] = diff

for node in G:
    if G.out_degree(node)!=0:
        G.nodes[node]['y']=-1
    

#fix unlabelled nodes to -1
for node in G:
    try:
        if G.nodes[node]['y']==-1:
            pass
    except Exception as e:
        G.nodes[node]['y']=-1

for node in G:
    try:
        feat = G.nodes[node]['x']
        pass
    except Exception as e:
        G.nodes[node]['x']=np.zeros((1024,),dtype=np.float32)

l = [G.subgraph(c) for c in nx.connected_components(nx.to_undirected(G))]


for i, tree in enumerate(l):
    for node in tree:
        if tree.nodes[node]['x'].dtype==np.float64:
            l[i].nodes[node]['x']=np.zeros((1024,),dtype=np.float32)

for i, tree in enumerate(l):
    f = False
    for node in tree:
        if tree.out_degree(node)==0:
            if tree.nodes[node]['y']==-1:
                f=True
                break
    if f:
        for node in tree:
            G.remove_node(node)

l = [G.subgraph(c) for c in nx.connected_components(nx.to_undirected(G))]

i=0
t = len(l)
k=0
for tree in l:
    for node in tree:
        if tree.out_degree(node)==0:
            if (i/t)<=0.7:
                l[k].nodes[node]['train_mask'] = 1
                l[k].nodes[node]['val_mask'] = 0                
                l[k].nodes[node]['test_mask'] = 0
            elif (i/t)<=0.8:
                l[k].nodes[node]['train_mask'] = 0
                l[k].nodes[node]['val_mask'] = 1                
                l[k].nodes[node]['test_mask'] = 0
            else:
                l[k].nodes[node]['train_mask'] = 0
                l[k].nodes[node]['val_mask'] = 0                
                l[k].nodes[node]['test_mask'] = 1
            i+=1
        else:
            l[k].nodes[node]['train_mask'] = 0
            l[k].nodes[node]['val_mask'] = 0                
            l[k].nodes[node]['test_mask'] = 0
    k+=1

attrs = ['x', 'y', 'del_t', 'train_mask', 'val_mask', 'test_mask']
trees = [dgl.from_networkx(nx_t,node_attrs=attrs) for nx_t in l]

with open("./pheme_graph.pkl",'wb') as f:
    pickle.dump(trees,f)
