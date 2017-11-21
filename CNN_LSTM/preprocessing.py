from __future__ import print_function
import os
import pandas as pd
from build_vocab import build_vocab

''' this file's dir should be sibling to data/
    eliminate non informative positions
    one-hot encode and output each RRM sequence into csv
    takes ~30 mins for 1e6 sequences'''


def txt_to_csv(raw_txt_path, sep=None):
    """parses txt file or fasta file into csv
    info_positions: list of positions populated beyond a threshold"""

    print('Parsing sequence input file...')

    dic = dict()
    name = None
    seq = None
    with open(raw_txt_path) as RRM:
       for i, line in enumerate(RRM):
            if '#' in line:
                pass
            else:
                name, seq = line.split(sep)
                name = name.replace('/', '_') # to distinguish from directory
                name = name.replace('>', '')
                seq = seq.replace('\n', '')
                # separator down the line
                dic.update([(name, seq)])

    df = pd.DataFrame(list(map(lambda x: list(x.upper()), list(dic.values()))), index=dic.keys())
    return df

def informative_positions(df, processed_RRM_path, top_n=82, placeholder='-'):
    """rid of excessive placeholders, 
    keeping top_n most populated positions
    for pfam dataset, this is equivalent to keeping populated rate @.001 """

    print('Extracting top %d most populated positions...'%top_n)

    populate_rate = df.applymap(lambda x: x != placeholder).sum(axis=0) / \
                    df.applymap(lambda x: 1 if x else 0).sum(axis=0)
    positions_to_keep = [i for i, rate in enumerate(populate_rate) \
                         if rate in sorted(populate_rate, reverse=True)[:top_n]]
    informative_values = list(map(lambda x: ['<start>'] + x.tolist() + ['<end>'], 
                                  df[positions_to_keep].values))
    df1 = pd.DataFrame(informative_values, index=df.index)
    df1.to_csv(processed_RRM_path)
    return df1

def preprocess(preprocessed, RRM_path, output_path, sep=' ', vocab=None):
    assert os.path.isfile(RRM_path), 'input RRM path: %s not found!' %(RRM_path)
    df = pd.read_csv(RRM_path, index_col=0)
    if not preprocessed:
        df = txt_to_csv(RRM_path)
        df= informative_positions(df, processed_RRM_path=output_path)
    if not vocab:
        vocab = build_vocab(df)
        return vocab, df
    else:
        return df
