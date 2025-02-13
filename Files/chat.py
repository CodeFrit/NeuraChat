import random
import json
import torch
from model import FFModel
import util as u
import os
import platform

with open("Files/intents.json",'r') as f:
    inents = json.load(f)

FILE =  "data.pth"
data = torch.load(FILE)

num_inputs=data["num_inputs"]
num_class=data["num_class"]
hidden_size=data["hidden_size"]
my_words = data["my_words"]
tags = data["tags"]
model_state=data["model_state"] #0.85
answer_threshold = 0.71

model = FFModel(num_inputs,num_class,hidden_size)
model.load_state_dict(model_state)
model.eval()

print("Hello! Let's talk! Say 'quit' to quit and 'clear' to clear the chat.")

def chat(username = "Neura",repl="",targ="") -> int:
    sentence = input("You: ").lower()
    sentence = sentence.replace(repl,targ)
    if sentence == "quit":
        print("Quiting...")
        return 1
    if sentence == "clear":
        cls()
        print("Hello! Let's talk! Say 'quit' to quit and 'clear' to clear the chat.")
        return 2
    sentence=u.tokenize(sentence)
    X = u.word_bag(sentence,my_words)
    X = X.reshape(1,-1) #1,samples
    X = torch.from_numpy(X)

    output = model(X)
    _,pred = torch.max(output,1)
    tag = tags[pred.item()]
    prob = torch.softmax(output,1)[0][pred.item()]

    if prob > answer_threshold:
        for inp in inents["intents"]:
            if(tag==inp["tag"]):
                print(f"{username}: {random.choice(inp['responses'])}")
    else:
        print(f"{username}: Sorry, I don't know how answer that!")
    return 0

def cls():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear') #other


while True:
    if(chat("Neura","motivate me","motivate")==1):
        break