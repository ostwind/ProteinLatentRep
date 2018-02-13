import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO


# # Similarity regression:

# # RWA^T = Y
# # R:  low-level embeddings
# # W: weight matrix to learn
# # A: RNA/protein sequences to match
# # Y: known matches
## following class returns LHS 

class SimilarityRegression(nn.Module):
    def __init__(self, emb_dim, rna_dim):
        super(AfinityRegression, self).__init__()
        # want to learn weights after applying to embedding layer (First do RW , then RW A^T)
        self.lin1 = nn.Linear(emb_dim, rna_dim)
        self.lin2 = nn.Linear(rna_dim, emb_dim)
        self.lin3 = nn.Linear(emb_dim, emb_dim)
    
    def init_weights(self):
        for x in self.lin1.parameters():
            x.data = x.data.normal_(0.0, 0.02)
        return None
    
    def forward_old(self, batch_size, embs, rna_samples):
        self.batch_size = batch_size
        print("a: ", rna_samples.size())
        x = self.lin2(rna_samples)
        print("b: ", x.size())
        x = torch.matmul(embs, x.t())
        print("c: ", x.size())
        return x
    
    
    def forward(self, batch_size, embs, rna_samples):
        self.batch_size = batch_size
        x = self.lin3(embs)
        x = torch.matmul(x, rna_samples)
        return x
    
    