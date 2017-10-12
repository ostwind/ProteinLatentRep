''' first transform raw text into csv
    transform raw text into csv again, but eliminating non informative positions
    one-hot encode and pickle
    pytorch loader (in loader.py) simply unpacks then yields samples
'''
from torchtext import data
#import torchtext.data
import torch.utils.data
import os
import pandas as pd
from preprocessing import *

def loader(data_pickle_path = "data.p", name_pickle_path = "names.p"):
    #assert os.path.isfile(data_pickle_path) and os.path.isfile(
    #name_pickle_path), '%s or %s not valid ' %(data_pickle_path, name_pickle_path)
    if not os.path.isfile(data_pickle_path) or not os.path.isfile(data_pickle_path):
        print('data and label pickles missing, preprocessing')
        preprocess()

    dataset = pickle.load(open(data_pickle_path, "rb"))
    name_list = pickle.load(open(name_pickle_path, "rb"))
    #print(dataset.shape, name_list.shape)

    source = TensorDataset(torch.from_numpy(dataset),
    torch.randn(dataset.shape[0])) # labels to be inserted here

    loader = DataLoader(source, batch_size = 64, shuffle = True)
    return loader

for data, label in loader():
    print(data.shape, label.shape)
