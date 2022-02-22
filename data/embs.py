import pandas as pd
import transformers
from transformers import BertModel,BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm
import torch
import gc
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./df_pheme_for_enc.pkl","rb") as f:
  df=pickle.load(f)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large').to(device)

df["enc"] = [np.array([0])]*len(df)

for idx,row in tqdm(df.iterrows(),total=len(df),position=0,leave=True):
  try:
    tweet = row["text"]
    input_ids = torch.tensor(tokenizer.encode(tweet)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb=last_hidden_states[0][0]
    emb=emb.to('cpu').detach().numpy()
    df.at[idx,"enc"]=emb
    if(idx%10==0 and idx>0):
      with open("./pheme_roberta_1024.pkl","wb") as f:
        pickle.dump(df,f)
    del outputs      
    del last_hidden_states
    gc.collect()

  except KeyboardInterrupt:
    break
  except:
    pass
