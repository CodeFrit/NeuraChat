import nltk
from nltk.stem.porter import PorterStemmer as pstm
import numpy as np

stemmer = pstm()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def word_bag(sentence,words):
    sentence = [stem(w) for w in sentence]
    ret = []
    for w in words:
        if w in sentence:
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.array(object=ret,dtype=np.float32)