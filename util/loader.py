''' loads pickles produced via preprocessing.py
    instantiates a pytorch loader according to input parameters
'''
from torch.utils.data import TensorDataset, DataLoader
from util.preprocessing import *
import numpy as np

def loader( batch_size = 64, shuffle = True, train_portion = 0.9, 
data_pickle_path = "./data/data.p", name_pickle_path = "./data/names.p"):
    if not os.path.isfile(data_pickle_path) or not os.path.isfile(name_pickle_path):
        print('data or label pickles missing, preprocessing from raw file @ ../data/PF00076_rp55.txt')
        preprocess(raw_txt_path = './data/combineddata.fasta')

    dataset = pickle.load(open(data_pickle_path, "rb"))
    name_list = pickle.load(open(name_pickle_path, "rb"))
    name_indices  = np.array(range(len(name_list)))
    
    num_samples = dataset.shape[0]
    uniform_sampling = np.random.random_sample((num_samples,))
    #train_indices = train_indices[train_indices < train_portion ]

    train_dataset = dataset[ uniform_sampling < train_portion]
    train_labels = name_indices[  uniform_sampling < train_portion]

    valid_dataset = dataset[ uniform_sampling >= train_portion ]
    valid_labels = name_indices[  uniform_sampling >= train_portion]
    #print(dataset.shape, name_list.shape)
    source = TensorDataset(torch.from_numpy(train_dataset),
    torch.from_numpy(train_labels)) 

    train_loader = DataLoader(
    source, batch_size = batch_size, shuffle = shuffle)

    valid_source = TensorDataset(torch.from_numpy(valid_dataset),
    torch.from_numpy(valid_labels)) 

    valid_loader = DataLoader(
    valid_source, batch_size = batch_size, shuffle = shuffle)
    
    return train_loader, valid_loader


if __name__ == "__main__":
    preprocess()
    # for data, label in loader():
    #    print(data, label)
    