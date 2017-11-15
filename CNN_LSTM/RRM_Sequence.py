import numpy as np
import pandas as pd
import torch
from build_vocab import build_vocab
from torch.utils.data import Dataset, DataLoader

class RRM_Sequence(Dataset):

    """RRM dataset without one-hot encoding"""

    def __init__(self, df, vocab):
        """
        Args:
            info_path (string): path for filtered RRM sequence csv file
            names (list of strings): RRM sequences to include
        """
        super(RRM_Sequence).__init__()
        self.names = df.index # this could potentially be replaced by gene ID's 
        self.df = df
        self.vocab = vocab
            

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = self.names[idx]
        rrm_aligned = [self.vocab(word) for word in \
        self.df.loc[rrm_name, :].values.tolist()]# list of integers
        rrm_unaligned = [integer for integer in rrm_aligned if integer != self.vocab('-')]
        return rrm_name, torch.Tensor(rrm_aligned), torch.Tensor(rrm_unaligned)

def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=True)
    names, rrms_aligned, rrms_unaligned = zip(*data)
    rrms_aligned = torch.stack(rrms_aligned, 0).long()

    lengths = [len(rrm) for rrm in rrms_unaligned]
    unaligned = torch.zeros(len(rrms_unaligned), max(lengths)).long()
    for i, rrm in enumerate(rrms_unaligned):
        end = lengths[i]
        unaligned[i, :end] = rrm[:end]
    return names, rrms_aligned, unaligned, lengths     







