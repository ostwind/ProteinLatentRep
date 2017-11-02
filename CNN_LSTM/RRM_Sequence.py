import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RRM_Sequence(Dataset):
    """RRM dataset without one-hot encoding"""

    def __init__(self, indices, info_path = '../data/rrm_rp55_info.csv', transform=None):
        """
        Args:
            info_path (string): path for filtered RRM sequence csv file
            names (list of strings): RRM sequences to include
            transform (callable, optional): optional transform to be applied
        """
        super(RRM_Sequence).__init__()
        self.names = indices
        self.csv_file = info_path
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = self.names[idx]
        
        print(rrm_name) # DEBUGGING
        
        rrm = pd.read_csv(self.csv_file, index_col=0)

        print(rrm.index.values) # DEBUGGING
        print(rrm['A0A1B7NG40.1_71-102']) # DEBUGGING
        BOOM # DEBUGGING
        
        rrm = rrm[rrm_name]
        
        rrm = torch.from_numpy(rrm).contiguous().float()
        sample = {'name': rrm_name, 'seq': rrm}

        if self.transform:
            sample = self.transform(sample)

        return sample
