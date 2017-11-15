"""creates 3 index csvs for train/val/test data
7:2:1"""

import pandas as pd
import numpy as np

np.random.seed(0)
ind_df = pd.read_csv('../data/combined_data.txt') #('../data/rrm_rp55.csv')
msk = np.random.randn(ind_df.shape[0]) < .7
ind_df[msk].iloc[:,0].to_csv('../data/train_index.csv', 
	index=False, header=False)
test = ind_df[~msk]
test_msk = np.random.randn(test.shape[0]) < 1/3
test[test_msk].iloc[:,0].to_csv('../data/test_index.csv', 
	index=False, header=False)
test[~test_msk].iloc[:,0].to_csv('../data/val_index.csv', 
	index=False, header=False)
