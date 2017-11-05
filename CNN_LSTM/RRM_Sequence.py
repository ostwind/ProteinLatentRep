import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset #, DataLoader

class RRM_Dataset(Dataset):
    """One hot encoded RRM dataset"""

    def __init__(self, indices, root_dir='../data', transform=None):
        """
        Args:
            csv_file (string): directory of csv file with all RRM sequences.
            root_dir (string): directory with all the csv file for
            individual RRM csv.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(RRM_Dataset).__init__()
        self.root_dir = root_dir
        self.names = indices
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = self.names[idx]
        rrm_path = os.path.join(self.root_dir, self.names[idx]+'.csv')
        rrm = pd.read_csv(rrm_path, index_col=0).as_matrix().astype('float')
        rrm = torch.from_numpy(rrm).contiguous().float()
        sample = {'name': rrm_name, 'seq': rrm}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RRM_AlignedSequence(Dataset):
    """RRM dataset without one-hot encoding and with alignment"""

    def __init__(self, indices, info_path = '../data/rrm_rp55_info.csv'):
        """
        Args:
            info_path (string): path for filtered RRM sequence csv file
            names (list of strings): RRM sequences to include
        """
        super(RRM_AlignedSequence).__init__()
        self.names = indices
        self.df = pd.read_csv(info_path, index_col=0)

    def __len__(self):
        # this could potentially be replaced by gene ID's 
        # @Millie
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = self.names[idx]
        rrm = ''.join(self.df.loc[rrm_name, :].values)
        rrm = '<start>' + rrm + '<end>'
        sample = {'name': rrm_name, 'seq': rrm}
        return sample

class RRM_OriginalSequence(Dataset):
    """RRM dataset without one-hot encoding and without alignment"""

    def __init__(self, indices, raw_txt_path = '../data/PF00076_rp55-2.txt'):
        """
        Args:
            raw_txt_path (string): path for original RRM sequence txt file
            names (list of strings): RRM sequences to include
        """
        super(RRM_OriginalSequence).__init__()
        self.names = indices
        self.df = self._parse_input(raw_txt_path)

    def __len__(self):
        # this could potentially be replaced by gene ID's 
        # @Millie
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = self.names[idx]
        rrm = ''.join(self.df.loc[rrm_name, :].values)
        rrm = '<start>' + rrm + '<end>'
        sample = {'name': rrm_name, 'seq': rrm}
        return sample

    def _parse_input(self, raw_txt_path, sep=None):

        """parses txt file or fasta file into csv
        info_positions: list of positions populated beyond a threshold"""

        print('Parsing sequence input file...')

        dic = dict()
        with open(raw_txt_path) as RRM:
            for i, line in enumerate(RRM):
                if '#' in line:
                    pass
                else:
                    name, seq = line.split(sep)
                    name = name.replace('/', '_') # to distinguish from directory
                    # separator down the line
                    dic.update([(name, seq)])

        df = pd.DataFrame(list(dic.values()), index=dic.keys())

        return df 
