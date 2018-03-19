''' loads pickles produced via preprocessing.py
    instantiates a pytorch loader according to input parameters
'''
from torch.utils.data import TensorDataset,Dataset, DataLoader
from util.preprocessing import *
import numpy as np

#denoising pass: clean and noisy unlabeled data
#backtranslation pass: clean unlabeled data   
#supervised pass: labeled data with vector to check

#TODO UNALIGNED SEQUENCE
def loader( alignment, batch_size = 64, shuffle = True, train_portion = 0.95, seq_format = False, ):
    np.random.seed(0)

    data_pickle_path = "./data/%s/data.p" %(alignment)
    name_pickle_path = "./data/%s/names.p" %(alignment)

    if not os.path.isfile(data_pickle_path) or not os.path.isfile(name_pickle_path):
        print('data or label pickles missing, preprocessing from raw file @ ../data/PF00076_rp55.txt')
        preprocess(raw_txt_path = './data/combineddata.fasta')

    if seq_format:
        data_pickle_path = "./data/data_seq_format.p"
        name_pickle_path = "./data/names_seq_format.p"

    dataset = pickle.load(open(data_pickle_path, "rb"))
    num_samples = dataset.shape[0]
    uniform_sampling = np.random.random_sample((num_samples,))
    
    #dataset = torch.from_numpy(dataset)
    name_list = pickle.load(open(name_pickle_path, "rb"))
    name_indices  = np.array(name_list)
    #print(dataset.shape, name_indices.shape )

    train_dataset = dataset[ uniform_sampling < train_portion]
    train_labels = name_indices[  uniform_sampling < train_portion]
    valid_dataset = dataset[ uniform_sampling >= train_portion ]
    valid_labels = name_indices[  uniform_sampling >= train_portion]

    #print(len(train_dataset[0]), len(train_dataset[4]))
    
    valid_source = TensorDataset(torch.from_numpy(valid_dataset),
    torch.from_numpy(valid_labels)) 

    valid_loader = DataLoader(
    valid_source, batch_size = batch_size, shuffle = shuffle, drop_last=False)
        
    if train_portion == 0:
        return 0, valid_loader

    source = TensorDataset(torch.from_numpy(train_dataset), 
    torch.from_numpy(train_labels)) 

    train_loader = DataLoader(
    source, batch_size = batch_size, shuffle = shuffle, drop_last=False)

    
    return train_loader, valid_loader
    