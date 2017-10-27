''' run file from project directory e.g. python Seq2Vec.train.py
    this file trains a Seq2Vec model then tsne projects the representation
'''
import sys, os

from model import Seq2Vec
import numpy as np 
from util.tsne import tsne_plot
import matplotlib.pyplot as plt

fasta_input = './data/RRM_55.fasta' 
data_dir = './data/rrm_processed/' 


if __name__ == '__main__':
    model = Seq2Vec(None, fasta_input, data_dir)
    
    # example latent rep for a sequence
    #example_vect = model.vect_rep( 'sp|P04637|P53_HUMAN_0' )
    #print(example_vect)

    # all ids
    # name_ordering = list(model.all_ids())

    name_ordering = list(model.all_ids())
    #print(sorted(name_ordering))
    #exit()
    
    # filtering by first letter of sequence names 
    starting_letters = ['C', 'H', 'E']
    name_ordering = [ name for name in name_ordering if name[0] in starting_letters]
    #labels = ['H' if 'HUMAN' in name else 'A' for name in name_ordering ]

    representation = []
    for name in name_ordering:
        representation.append(model.vect_rep( name ))

    representation = np.array(representation)

    labels = [ name[0] for name in name_ordering ]

    tsne_plot('seq2vec_%s' %("".join(starting_letters)),
    labels, representation, take_first_n = 1000, n_iter = 2000)      
    
    
    