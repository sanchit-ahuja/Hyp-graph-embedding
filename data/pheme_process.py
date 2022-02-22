import os
import json
import numpy as np
import dgl
import torch
import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm

def convert_annotations(annotation, string = True):
    if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
        if int(annotation['misinformation'])==0 and int(annotation['true'])==1 :
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==0 :
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==1:
            label = None
        else:
            label =  None            
    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        if int(annotation['misinformation'])==1:
            if string:
                label = "false"
            else:
                label = 0
        else:
            label = 0
     
    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        label = None
    else:
        label = None
           
    return label

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

dirs = ['non-rumours','rumours']
data_dir = './pheme/all-rnr-annotated-threads'
dirs2 = ['reactions','source-tweets']

df = pd.DataFrame(columns=['tweet_id','parent_id','text','label','date','lang','dataset'])

sets = list(listdir_nohidden(data_dir))
for dset in sets:
    print(dset)
    ds = dset.split('-')[0]
    for i in dirs:
        files_dir = os.path.join(data_dir,dset,i)
        folders = list(listdir_nohidden(files_dir))
        for folder in tqdm(folders):
            annotation = json.load(open(os.path.join(files_dir,folder,'annotation.json')))
            label = convert_annotations(annotation,False)
            for j in dirs2:
                folder_path = os.path.join(files_dir,folder,j)
                files = listdir_nohidden(folder_path)
                for file in files: 
                    dic={}
                    file_path = os.path.join(folder_path,file)
                    f = open(file_path)
                    tweet_obj = json.load(f)
                    dic['tweet_id'] = str(tweet_obj.get('id'))
                    dic['parent_id'] = str(tweet_obj.get('in_reply_to_status_id'))
                    dic['text'] = str(tweet_obj.get('text'))
                    dic['lang'] = str(tweet_obj.get('lang'))
                    dic['dataset'] = ds
                    dic['label'] = label
                    dic['date'] = str(tweet_obj.get('created_at'))
                    df = df.append(dic,ignore_index=True)                      
    print(dset," done")

df = df.drop_duplicates('tweet_id').reset_index(drop=True)
df["label"] = df["label"].fillna(value=-1) #replies
df_new = df[df['lang']=='en']
with open("df_pheme_for_enc.pkl",'wb') as f:
    pickle.dump(df_new,f)
