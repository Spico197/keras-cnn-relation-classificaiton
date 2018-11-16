from utils import load_data
from config import config
from word2vec import get_vectorid_by_word
import numpy as np
import pickle

with open("preprocessed_data/data.pkl", 'rb') as file:
    data = pickle.load(file)

for key, item in data['train_set'].items():
    try:
        print(key, " - ", item.dtype)
    except:
        print(key, "-",type(item))