''' this file's dir should be sibling to data/
    eliminate non informative positions
    one-hot encode and output each RRM sequence into csv
    takes ~30 mins for 1e6 sequences
    TODO: add argparse, remove hard-coded values
'''
import os
import pandas as pd
import numpy as np
import argparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def txt_to_csv(raw_txt_path, csv_path, sep=None):

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

    df = pd.DataFrame(list(map(lambda x: list(x), list(dic.values()))), index=dic.keys())
    df.to_csv(csv_path)
    return df


def informative_positions(df, top_n=164, placeholder='-'):

    """rid of excessive placeholders, 
    keeping top_n most populated positions
    for pfam dataset, this is equivalent to keeping populated rate @.001 """

    print('Extracting top %d most populated positions...'%top_n)

    populate_rate = df.applymap(lambda x: x != placeholder).sum(axis=0)/\
    df.applymap(lambda x: 1 if x else 0).sum(axis=0)
    positions_to_keep = [i for i, rate in enumerate(populate_rate) \
    if rate in sorted(populate_rate, reverse=True)[:top_n]]

    return df[positions_to_keep]

def encode_output(df, onehot_path, label_encoder_path, scheme='onehot_encoder', root_dir='../data'):

    """encodes labels either into one-hot or integers, 
    output one csv for each RRM"""

    print('Encoding sequences...')

    vocab = np.unique(df)
    label_encoder = LabelEncoder()
    label_encoder.fit(vocab)
    pickle.dump(label_encoder, open(label_encoder_path, "wb" ))
    encoded = label_encoder.transform(df.values.ravel())\
    .reshape(df.shape)
    if scheme == 'onehot_encoder':
        onehot_encoder = OneHotEncoder(sparse=False)
        encoded = encoded[:, :, np.newaxis]
        onehot_encoder.fit(np.arange(len(vocab)).reshape(-1, 1))
        pickle.dump(onehot_encoder, open(onehot_path, "wb" ))
    for i, rrm in enumerate(map(lambda x: onehot_encoder.transform(x).tolist(), 
        encoded)):
        if i%500 == 0:
            print('\t%d/%d...'%(i, df.shape[0]))
        if i == df.shape[0] - 1:
            print('Done!')
        rrm = pd.DataFrame(np.array(rrm))
        rrm.to_csv(os.path.join(root_dir, df.index[i] + '.csv'))

def preprocess(raw_txt_path = '../data/PF00076_rp55.txt'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_txt_path', type=str, default='../data/PF00076_rp55.txt', 
        help='path for aligned RRM input txt file')
    parser.add_argument('--csv_path', type=str, default='../data/rrm_rp55.csv', 
        help='path for  RRM sequence output csv file')
    parser.add_argument('--top_n', type=int, default=82, 
        help='include top n most populated positions')
    parser.add_argument('--encoder_path', type=str, default='../data/onehot.p', 
        help='path for one-hot encoder')
    parser.add_argument('--label_encoder_path', type=str, default='../data/label_encoder.p', 
        help='path for one-hot encoder')
    parser.add_argument('--encoded_output_path', type=str, default='../data', 
        help='path for one-hot encoded individual RRM output csv files')

    args = parser.parse_args()
    raw_txt_path = args.raw_txt_path
    csv_path = args.csv_path 
    N = args.top_n
    onehot_path  = args.encoder_path
    label_encoder_path = args.label_encoder_path
    encoded_output_path = args.encoded_output_path
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)

    df = txt_to_csv(raw_txt_path, csv_path) # first convert to csv
    #filter empty positions
    df = informative_positions(df, top_n=N)
    encode_output(df, onehot_path, label_encoder_path, root_dir=encoded_output_path)

if __name__ == "__main__":
    preprocess()
