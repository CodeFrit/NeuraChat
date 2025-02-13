import json
import util as u
import numpy as np
from torch.utils.data import Dataset,DataLoader
from model import FFModel
import torch.nn as nn
import torch.optim as optim
import torch

with open("Files/intents.json",'r') as f:
    inents = json.load(f)

my_words = []
tags = []
xy = []

for inp in inents['intents']:
    tag = inp['tag']
    tags.append(tag)
    for pat in inp['patterns']:
        wort = u.tokenize(pat)
        my_words.extend(wort)
        xy.append((wort,tag))

ignore = ["?","!",".",","]
my_words = [u.stem(w) for w in my_words if w not in ignore]
my_words = sorted(set(my_words))
tags = sorted(set(tags))

x_trn = []
y_trn = []

for (pat,tag) in xy:
   bag = u.word_bag(pat,my_words)
   x_trn.append(bag) 

   label = tags.index(tag)
   y_trn.append(label) #using CEL so index = class

x_trn = np.array(x_trn)
y_trn = np.array(y_trn)

class ChatDat(Dataset):
    def __init__(self):
        self.n_sam = len(x_trn)
        self.x_dat = torch.from_numpy(x_trn)
        self.y_dat = torch.from_numpy(y_trn).long()

    def __getitem__(self, index):
        return self.x_dat[index],self.y_dat[index]
    
    def __len__ (self):
        return self.n_sam
    
#params
batch_size = 4
num_inputs=len(x_trn[0])
num_class=len(tags)
hidden_size=10
num_epochs=900
lrt = 0.008

dat_set = ChatDat()
train_load = DataLoader(dataset=dat_set,batch_size=batch_size,shuffle=True,num_workers=0)

model = FFModel(num_inputs,num_class,hidden_size)
loss = nn.CrossEntropyLoss()
optimezr = optim.Adam(params=model.parameters(),lr=lrt)

for e in range(num_epochs):
    for (xd,yd) in train_load:
        y_pred = model(xd)
        l = loss(y_pred,yd)
        optimezr.zero_grad()
        l.backward()
        optimezr.step()
    if(e%100==0):
        print(f"epoch: {e}/{num_epochs} loss: {l.item()}")

print(f"epoch: {num_epochs}/{num_epochs} loss: {l.item()}")

#saving
model_dat = {
    "model_state":model.state_dict(),
    "num_inputs":num_inputs,
    "num_class":num_class,
    "hidden_size":hidden_size,
    "my_words":my_words,
    "tags":tags
}

FILE =  "data.pth"
torch.save(model_dat,FILE)

print(f"Done, saved to {FILE}")