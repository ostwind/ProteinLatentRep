''' this file's dir should be sibling to data/
    eliminate non informative positions
    one-hot encode and output each RRM sequence into csv
    takes ~30 mins for 1e6 sequences
    TODO: add argparse, remove hard-coded values
'''
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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

def encode_output(df, scheme='onehot_encoder', root_dir='../data'):

    """encodes labels either into one-hot or integers, 
    output one csv for each RRM"""

    print('Encoding sequences...')

    vocab = np.unique(df)
    label_encoder = LabelEncoder()
    label_encoder.fit(vocab)
    encoded = label_encoder.transform(df.values.ravel())\
    .reshape(df.shape)
    if scheme == 'onehot_encoder':
        onehot_encoder = OneHotEncoder(sparse=False)
        encoded = encoded[:, :, np.newaxis]
        onehot_encoder.fit(np.arange(len(vocab)).reshape(-1, 1))
    for i, rrm in enumerate(map(lambda x: onehot_encoder.transform(x).tolist(), 
        encoded)):
        if i%500 == 0:
            print('\t%d/%d...'%(i, df.shape[0]))
        if i == df.shape[0] - 1:
            print('Done!')
        rrm = pd.DataFrame(np.array(rrm))
        rrm.to_csv(os.path.join(root_dir, df.index[i] + '.csv'))

def preprocess(raw_txt_path = '../data/PF00076_rp55.txt'):

    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)

    csv_path = '../data/rrm_rp55.csv' # non encoded csv file

    # cvs file after encoding

    df = txt_to_csv(raw_txt_path, csv_path) # first convert to csv

    #filter empty positions
    df = informative_positions(df)

    # import matplotlib.pyplot as plt
    # plt.hist(informative_positions(csv_path))
    # plt.show()

    # encode data and output csv file for each rrm
    encode_output(df)

if __name__ == "__main__":
    preprocess()
