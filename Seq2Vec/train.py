''' run file from project directory e.g. python Seq2Vec.train.py
    this file trains a Seq2Vec model then tsne projects the representation
'''
import sys, os
parent_dir = os.path.abspath(__file__ + "/../../")
sys.path.append(parent_dir)

from util.IO_tools import write_to_csv
from util.tsne import pca

from model import Seq2Vec
import numpy as np 
from util.tsne import plot
import matplotlib.pyplot as plt

from util.IO_tools import protein_rna_map

fasta_input = './data/RRM55_with_labeled.fasta' 
data_dir = './data/rrm_processed/' 

if __name__ == '__main__':
    model = Seq2Vec(None, fasta_input, data_dir)
    
    # example latent rep for a sequence
    #example_vect = model.vect_rep( 'sp|P04637|P53_HUMAN_0' )
    #print(example_vect)

    # all ids
    # name_ordering = list(model.all_ids())

    
    name_ordering = list(model.all_ids())
    #################################################
    # dictionary = protein_rna_map()
    

    representation = []

    for rrm_index, rrm in enumerate(name_ordering):
        representation.append(model.vect_rep(rrm))
    ##########################################################


    representation = np.array(representation)
    print(representation.shape)
    for dim in [64, 128, 200]:
        representation_lr = pca(representation, pca_dim = dim) 

        representation_path = './AffReg/Seq2Vec_%sDim.csv' %(dim)

        write_to_csv(name_ordering, representation_lr, representation_path)
        


    #48633/58510 were found in dictionary
    # rrm_found_in_dict = []
    # rna = []
    # for protein_index, protein in enumerate(name_ordering):
    #     ide = protein.split('.')[0]
    #     if ide in dictionary.keys():
    #         if '_' in dictionary[ide]: 
    #             rna.append(dictionary[ide].split('_')[0])
    #         else:
    #             rna.append(dictionary[ide])
    

    # from collections import Counter
    # tabulated = Counter(rna)
    # common_rnas = [r for r in rna if tabulated[r] > 0]
    
    # representation = []
    # label = []
    # for protein_index, protein in enumerate(name_ordering):
    #     ide = protein.split('.')[0]
    #     if ide in dictionary.keys():
    #         if dictionary[ide] in common_rnas:
    #             #print(dictionary[ide])
    #             representation.append(model.vect_rep(protein))
    #             label.append(dictionary[ide].split('_')[0])

    # sort_both = zip(label, representation)
    # sort_both = sorted(sort_both, key=lambda t: t[0]) 
    # label, representation = [pair[0] for pair in sort_both], [pair[1] for pair in sort_both]
    # representation = np.array(representation)

    # from util.tsne import plotly_scatter
    # plotly_scatter('seq2vec_20epochs_winsize30.png', label, representation)
    #plot('seq2vec_colored_by_rna.png', label, representation, )
    
    