import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RRM_AlignedSequence(Dataset):

    """RRM dataset without one-hot encoding"""

    def __init__(self, indices, info_path = '../data/rrm_rp55_info.csv'):
        """
        Args:
            info_path (string): path for filtered RRM sequence csv file
            names (list of strings): RRM sequences to include
        """
        super(RRM_Sequence).__init__()
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

    """RRM dataset without one-hot encoding"""

    def __init__(self, indices, raw_txt_path = '../data/PF00076_rp55-2.txt'):
        """
        Args:
            info_path (string): path for filtered RRM sequence csv file
            names (list of strings): RRM sequences to include
        """
        super(RRM_Sequence).__init__()
        self.names = indices
        self.df = _parse_input(raw_txt_path)

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









