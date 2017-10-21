''' loads pickles produced via preprocessing.py
    instantiates a pytorch loader according to input parameters
'''
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import *

def loader( batch_size = 64, shuffle = True,
data_pickle_path = "data.p", name_pickle_path = "names.p"):
    if not os.path.isfile(data_pickle_path) or not os.path.isfile(data_pickle_path):
        print('data or label pickles missing, preprocessing from raw file @ ../data/PF00076_rp55.txt')
        preprocess()

    dataset = pickle.load(open(data_pickle_path, "rb"))
    name_list = pickle.load(open(name_pickle_path, "rb"))
    #print(dataset.shape, name_list.shape)
    source = TensorDataset(torch.from_numpy(dataset),
    torch.randn(dataset.shape[0])) # labels to be inserted here

    loader = DataLoader(
    source, batch_size = batch_size, shuffle = shuffle)
    return loader

if __name__ == "__main__":
    for data, label in loader():
        print(data.shape, label.shape)