''' run file from project directory e.g. python Seq2Vec.train.py
    this file trains a Seq2Vec model then tsne projects the representation
'''
import sys, os

from model import Seq2Vec
import numpy as np 
from util.tsne import plot
import matplotlib.pyplot as plt

from util.IO_tools import protein_rna_map

fasta_input = './data/RRM_55_sample_position_filtered.fasta' 
data_dir = './data/rrm_processed/' 


if __name__ == '__main__':
    model = Seq2Vec(None, fasta_input, data_dir)
    
    # example latent rep for a sequence
    #example_vect = model.vect_rep( 'sp|P04637|P53_HUMAN_0' )
    #print(example_vect)

    # all ids
    # name_ordering = list(model.all_ids())

    name_ordering = list(model.all_ids())
    dictionary = protein_rna_map()
    
    #48633/58510 were found in dictionary
    rrm_found_in_dict = []
    rna = []
    for protein_index, protein in enumerate(name_ordering):
        ide = protein.split('.')[0]
        if ide in dictionary.keys():
            if '_' in dictionary[ide]: 
                rna.append(dictionary[ide].split('_')[0])
            else:
                rna.append(dictionary[ide])
    
    from collections import Counter
    tabulated = Counter(rna)
    common_rnas = [r for r in rna if tabulated[r] > 40]
    
    representation = []
    label = []
    for protein_index, protein in enumerate(name_ordering):
        ide = protein.split('.')[0]
        if ide in dictionary.keys():
            if dictionary[ide] in common_rnas:
                #print(dictionary[ide])
                representation.append(model.vect_rep(protein))
                label.append(dictionary[ide].split('_')[0])

    sort_both = zip(label, representation)
    sort_both = sorted(sort_both, key=lambda t: t[0]) 
    label, representation = [pair[0] for pair in sort_both], [pair[1] for pair in sort_both]
    representation = np.array(representation)

    plot('seq2vec_colored_by_rna.png', label, representation, )
    
    