''' this file's dir should be sibling to data dir

    transform raw text into csv
    eliminating non informative positions
    integer encode (optionally one-hot encode) and pickle
    pytorch loader (in loader.py) simply unpacks then yields samples
'''
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
import numpy as np

#TODO parse fasta, etc. formats beyond .txt
def txt_to_csv(raw_txt_path, csv_path, info_positions = None):
    with open(raw_txt_path) as f:
        content = f.readlines()

    keys = []
    vals = []
    for line in content:
        non_empty_strings = [s for s in line.split(' ') if s]
        keys.append(non_empty_strings[0])

        if not info_positions:
            vals.append(non_empty_strings[-1][:-2]) # elim /n char

        else:
            sequence = non_empty_strings[-1]
            sequence2 = []
            for ind in info_positions:
                sequence2.append( sequence[ind] )
            vals.append( "".join(sequence2) )

    df = pd.DataFrame({'keys': keys, 'vals': vals})
    df.to_csv(csv_path)

def informative_positions(csv_path):
    df = pd.read_csv(csv_path)
    seq_list = df['vals'].tolist()

    positions_to_keep = []
    for position in range(len(seq_list[0])):
        tally = dict()
        for seq in seq_list:
            if seq[position] in tally.keys():
                tally[seq[position]] += 1
            else:
                tally[seq[position]] = 0

        na_vals = 0
        true_vals = 0
        for key, value in tally.items():
            if '-' in key:
                na_vals += value
            else:
                true_vals += value

        percent = true_vals/(true_vals+na_vals)
        if percent > 0.01: # if  >1% of samples have non-empty value, keep position
            positions_to_keep.append(position)
    return positions_to_keep

def one_hot_pickle(csv_path):
    df = pd.read_csv(csv_path)
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

    pickle.dump( np.array(dataset), open( "data.p", "wb" ) )
    pickle.dump( np.array(name_list), open( "names.p", "wb" ) )

def preprocess(raw_txt_path = '../data/PF00076_rp55.txt'):
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)
    csv_path = '../data/rrm_rp55.csv'
    informative_csv_path = '../data/rrm_rp55_condensed.csv'

    txt_to_csv(raw_txt_path, csv_path) # first convert to csv

    #filter empty positions then re-write csv to informative_csv_path
    txt_to_csv(raw_txt_path, informative_csv_path, informative_positions(csv_path))

    # import matplotlib.pyplot as plt
    # plt.hist(informative_positions(csv_path))
    # plt.show()

    one_hot_pickle(informative_csv_path)
    os.remove(csv_path)
    os.remove(informative_csv_path)
