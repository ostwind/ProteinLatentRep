from __future__ import print_function
import numpy as np
"""creates 3 index csvs for train/val/test data
7:2:1"""
def train_test_split(df):
    np.random.seed(0)  # COMBINED DATA
    msk = np.random.rand(df.shape[0]) < .7
    df['index'] = df.index
    df[msk].loc[:,'index'].to_csv('../data/train_index.csv', index=False, header=False)
    test = df[~msk]
    test_msk = np.random.rand(test.shape[0]) < 1.0/3
    test[test_msk].loc[:,'index'].to_csv('../data/test_index.csv', index=False, header=False)
    test[~test_msk].loc[:,'index'].to_csv('../data/val_index.csv', index=False, header=False)
    df = df.drop('index', 1)
    return df