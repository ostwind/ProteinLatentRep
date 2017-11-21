""" gets encoded z space representation from saved model,
outputs tsne transformed visualization"""

# TODO add argparse for input file

import torch
import argparse
import pandas as pd
from preprocessing import txt_to_csv
from AutoEncoder import VAE
from torch.autograd import Variable
import sys
import os
sys.path.append(os.getcwd()+'/'+'../util')
from tsne import tsne_plot

parser = argparse.ArgumentParser(description='visualizes z space rep in 2d')
parser.add_argument('--letters-list', type=list, default=['Q', 'J'], 
    help='a list for letters of interest, each letter group in quotes')


def get_z(matrices, types):

	"""given <matrices>: a list of original one-hot encoded representation,
	and <types>: a list of letter combinations of interest,
	returns a list of transformed z-space representations"""

    for i, (matrix, typ) in enumerate(zip(matrices, types)):
        v = Variable(torch.from_numpy(matrix).contiguous().float())
        z, recon, mu, logvar = autoencoder(v)
        z = z.data.numpy()
        z = pd.DataFrame(z)
        z['label'] = pd.Series([typ]*z.shape[0])
        if i == 0:
            latent_sequence = z
        else:
            latent_sequence = pd.concat((latent_sequence, z))
    return latent_sequence

def latent_matrix(letters):

	""" extracts one-hot encoded reprentation for RRMs
	starting with <letters>""" 

    msk = [ind for ind in match.index if letters in ind[:len(letters)]]
    rrms = [ind.replace('/', '-') for ind in match.ix[msk, :].index]
    matrix = pd.read_csv('../data/' + rrms[0]+'.csv', index_col=0).as_matrix().astype('float').ravel()
    for name in rrms[1:]:
        rrm = pd.read_csv('../data/' + name +'.csv', index_col=0).as_matrix().astype('float').ravel()
        matrix = np.vstack((matrix, rrm))
    return matrix

def main():
	args = parser.parse_args()

	autoencoder = VAE()
	autoencoder.load_state_dict(torch.load('./model.pt'))

	raw_txt_path = '../data/PF00076_rp55.txt'
	csv_path = '../data/rrm_rp55.csv' 
	match = txt_to_csv(raw_txt_path, csv_path)
	# match is the df storing protein names and their sequence
	matrices = [latent_matrix(letters) for letters in args.letters_list]
	latent_sequence = get_z(matrices, args.letters_list)
	latent_sequence = latent_sequence.sample(frac=.1) 

	
	labels = latent_sequence['label']
	latent = latent_sequence.drop('label', 1)
	tsne_results = tsne_plot('AutoEncoder', labels, latent)


