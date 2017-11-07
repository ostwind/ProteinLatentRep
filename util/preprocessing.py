''' this file's dir should be sibling to data dir

    transform raw text into csv
    eliminating non informative positions
    integer encode (optionally one-hot encode) and pickle
    pytorch loader (in loader.py) simply unpacks then yields samples
'''
import os
import pandas as pd
from util.IO_tools import write_fasta 
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.tsne import hist

#TODO parse fasta, etc. formats beyond .txt
def txt_to_csv(raw_txt_path,  position_ind = None, sample_ind = None):
    keys = []
    vals = []
    with open(raw_txt_path) as RRM:
        for protein_index, protein in enumerate(RRM):
            # filtering by samples, skip if the sample indicator variable is 0 at current sample    
            if sample_ind:
                if not sample_ind[protein_index]:
                    continue
            
            non_empty_strings = [s for s in protein.split(' ') if s]
            keys.append(non_empty_strings[0].replace("/","_"))
            if not position_ind:
                vals.append(non_empty_strings[-1][:-2]) # elim /n char

            else:
                # filtering by positions
                sequence = non_empty_strings[-1]
                sequence_filtered = []
                for i in range(len(position_ind)):
                    if position_ind[i]: # 1/0 indicator bit
                        sequence_filtered.append( sequence[i] )
                vals.append( "".join(sequence_filtered) )

    if sample_ind: # if position + sample filtered, write fasta version for Seq2Vec
        write_fasta(vals, keys, fasta_name = './data/RRM_55_sample_position_filtered.fasta')
        print(len(vals), ' samples made it')

    df = pd.DataFrame({'keys': keys, 'vals': vals})
    
    # write df if it dosen't fit in RAM 
    return df

def _filter_positions(df, threshold = 0.01, plot=False):
    seq_list = df['vals'].tolist()
    keep_pos_ind = []
    position_occupancies = []

    for position in range(len(seq_list[0])):
        # string of all position-th symbols in every sequence
        aggregate_position_string = [ seq[position] for seq in seq_list ]
        non_blank_symbols = [ symbol for symbol in aggregate_position_string \
        if symbol != '-']
        information_percent = len(non_blank_symbols) / len(aggregate_position_string)
        
        position_occupancies.append(information_percent)

        if information_percent > threshold: # if  >1% of samples have non-empty value, keep position
            keep_pos_ind.append(1)
        else:
            keep_pos_ind.append(0)
    
    if plot:
        hist('Position Occupancies by Percentage', position_occupancies)

    return keep_pos_ind

def _filter_samples(df, plot = False):
    seq_list = df['vals'].tolist()
    sample_occupancies =[]
    keep_sample_ind = []
    for seq_ind, seq in enumerate(seq_list):

        percent_occupied = len([char for char in seq if char != '-'])/len(seq)
        sample_occupancies.append( percent_occupied )
        
        if percent_occupied > 0.8: # keep 58511/99932 = 0.585% proteins
            keep_sample_ind.append(1)
        else:
            keep_sample_ind.append(0)

    if plot:
        hist('Sample Occupancies by Percentage', sample_occupancies)

    return keep_sample_ind  

def one_hot_pickle(df2):
    seq_list = df2['vals'].tolist()
    name_list = df2['keys'].tolist()

    label_encoder = LabelEncoder()
    #onehot_encoder = OneHotEncoder(sparse=False)
    dataset = []
    for index, seq in enumerate(seq_list):
        values = np.array(list(seq))
        integer_encoded = label_encoder.fit_transform(values)
        #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        #print('integer encoded sequence shape: %s' %(integer_encoded.shape))
        dataset.append(integer_encoded)
        #print(integer_encoded)

    pickle.dump( np.array(dataset), open( "./data/data.p", "wb" ) )
    pickle.dump( np.array(name_list), open( "./data/names.p", "wb" ) )

def preprocess(raw_txt_path = './data/PF00076_rp55.fasta'):
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)
    
    df = txt_to_csv(raw_txt_path, ) # first convert to csv
    #filter empty positions then re-write csv to informative_csv_path
    
    position_ind = _filter_positions(df)
    df1 = txt_to_csv(raw_txt_path,  position_ind = position_ind)

    sample_ind = _filter_samples(df1)

    df2 = txt_to_csv(raw_txt_path, position_ind = position_ind,
    sample_ind = sample_ind,
    )
    one_hot_pickle(df2)