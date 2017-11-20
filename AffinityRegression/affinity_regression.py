import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO


# # affinity regression:

# # RWA^T = Y
# # R:  low-level embeddings
# # W: weight matrix to learn
# # A: RNA/protein sequences to match
# # Y: known matches
## following class returns LHS 

class AfinityRegression(nn.Module):
    def __init__(self, emb_dim, rna_dim):
        super(AfinityRegression, self).__init__()
        # want to learn weights after applying to embedding layer (First do RW , then RW A^T)
        self.lin1 = nn.Linear(emb_dim, rna_dim)
    
    def init_weights(self):
        return None
    
    def forward(self, batch_size, embs, rna_samples):
        self.batch_size = batch_size
        x = self.lin1(embs)
        x = torch.matmul(x, rna_samples)

        return x