"""Autoencoder for learning latent representation
Refs:
SparseAutoencoder: 
	https://discuss.pytorch.org/t/how-to-create-a-sparse-autoencoder-neural-network-with-pytorch/3703
VAE:
	https://github.com/pytorch/examples/blob/master/vae/main.py"""


import pandas as pd
import torch
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset


class RRM_Dataset(Dataset):

	"""One hot encoded RRM dataset"""

	def __init__(self, csv_file='../data/rrm_rp55.csv', root_dir='../data', transform=None):
        """
        Args:
            csv_file (string): directory of csv file with all RRM sequences.
            root_dir (string): directory with all the csv file for
            individual RRM csv.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
	    self.names = pd.read_csv(csv_file, index_col=0).index.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = os.path.join(self.root_dir, self.names[idx])
        rrm = pd.read_csv(rrm_name, index_col=0).as_matrix().astype('float')
        rrm = torch.from_numpy(rrm)
        sample = {'name': rrm_name, 'seq': rrm}

        if self.transform:
            sample = self.transform(sample)

        return sample


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__():

		self.lin1 = nn.Linear()












