''' this file's dir should be sibling to data dir

    transform raw text into csv
    eliminating non informative positions
    integer encode (optionally one-hot encode) and pickle
    pytorch loader (in loader.py) simply unpacks then yields samples
'''
import os
import pandas as pd
from IO_tools import write_fasta 
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
import numpy as np

#TODO parse fasta, etc. formats beyond .txt
def txt_to_csv(raw_txt_path, csv_path, info_positions = None):
    keys = []
    vals = []
    with open(raw_txt_path) as RRM:
        for line in RRM:
            non_empty_strings = [s for s in line.split(' ') if s]
            keys.append(non_empty_strings[0].replace("/","_"))

            if not info_positions:
                vals.append(non_empty_strings[-1][:-2]) # elim /n char

            else:
                sequence = non_empty_strings[-1]
                sequence_filtered = []
                for i in range(len(info_positions)):
                    if info_positions[i]: # 1/0 indicator bit
                        sequence_filtered.append( sequence[i] )
                vals.append( "".join(sequence_filtered) )

    write_fasta(vals, keys)

    df = pd.DataFrame({'keys': keys, 'vals': vals})
    # write df if it dosen't fit in RAM 
    return df

def informative_positions(df, threshold = 0.01):
    seq_list = df['vals'].tolist()
    keep_pos_ind = []
    for position in range(len(seq_list[0])):
        # string of all position-th symbols in every sequence
        aggregate_position_string = [ seq[position] for seq in seq_list ]
        non_blank_symbols = [ symbol for symbol in aggregate_position_string \
        if symbol != '-']
        information_percent = len(non_blank_symbols) / len(aggregate_position_string)
        
        if information_percent > threshold: # if  >1% of samples have non-empty value, keep position
            keep_pos_ind.append(1)
        else:
            keep_pos_ind.append(0)
    
    return keep_pos_ind

def one_hot_pickle(df):
    seq_list = df['vals'].tolist()
    name_list = df['keys'].tolist()

    label_encoder = LabelEncoder()
    #onehot_encoder = OneHotEncoder(sparse=False)
    dataset = []
    for index in range(len(seq_list)):
        values = np.array(list(seq_list[index]))
        integer_encoded = label_encoder.fit_transform(values)
        #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        #print('integer encoded sequence shape: %s' %(integer_encoded.shape))
        dataset.append(integer_encoded)
        print(integer_encoded)

    pickle.dump( np.array(dataset), open( "data.p", "wb" ) )
    pickle.dump( np.array(name_list), open( "names.p", "wb" ) )

def preprocess(raw_txt_path = '../data/PF00076_rp55.txt'):
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)
    csv_path = '../data/rrm_rp55.csv'
    #informative_csv_path = '../data/rrm_rp55_condensed.csv'

    df = txt_to_csv(raw_txt_path, csv_path) # first convert to csv

    #filter empty positions then re-write csv to informative_csv_path
    df = txt_to_csv(raw_txt_path, csv_path, informative_positions(df))

    one_hot_pickle(df)
    