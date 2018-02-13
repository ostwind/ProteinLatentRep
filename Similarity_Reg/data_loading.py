import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO

# making dataset for easy looping
class LowDimData(Dataset):
    def __init__(self, train, labels, transform=None):
        self.low_dim_embs = train
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.low_dim_embs)

    def __getitem__(self, idx):
        emb = self.low_dim_embs[idx]
        lab = self.labels[idx]
        
        sample = (emb, lab)
        if self.transform:
            sample = self.transform(sample)
        return sample
