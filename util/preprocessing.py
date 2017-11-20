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

def txt_to_csv(raw_txt_path,  position_ind = None, sample_ind = None):
    keys = []
    vals = []
    with open(raw_txt_path) as RRM:
        for line_index, line in enumerate(RRM):
            # filtering by samples, skip if the sample indicator variable is 0 at current sample    
            if sample_ind:
                if not sample_ind[line_index]:
                    continue
            
            non_empty_strings = [s for s in line.split(' ') if s]
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
    return df

def _filter_positions(df, threshold = 0.1, plot=True):
    seq_list = df['vals'].tolist()
    #print(seq_list[0], len(seq_list))
    
    keep_pos_ind, position_occupancies = [], []
    for position in range(len(seq_list[0])):
        # string of all position-th symbols in every sequence
        #print(seq_list[0])
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
        hist('Percentage of Non-Gap Symbols by Position', position_occupancies, )
    print(sum(keep_pos_ind),'/', len(keep_pos_ind), ' positions made it')
    return keep_pos_ind

def _filter_samples(df, threshold = 0.9, plot = True):
    seq_list = df['vals'].tolist()
    
    sample_occupancies, keep_sample_ind =[], []
    for seq_ind, seq in enumerate(seq_list):

        percent_occupied = len([char for char in seq if char != '-'])/len(seq)
        sample_occupancies.append( percent_occupied )
        
        # unlabeled only: keep 58511/99932 = 0.585% lines

        if percent_occupied > threshold: 
            keep_sample_ind.append(1)
        else:
            keep_sample_ind.append(0)

    if plot:
        hist('Percentage of Non-Gap Symbols by Sample', sample_occupancies)

    return keep_sample_ind  

def one_hot_pickle(df2):
    seq_list = df2['vals'].tolist()
    name_list = df2['keys'].tolist()

    label_encoder = LabelEncoder()
    #onehot_encoder = OneHotEncoder(sparse=False)
    label_encoder.fit(  list(set(''.join(seq_list))))
    #print((seq_list[0]))
    #print(label_encoder.inverse_transform(list(range(22))))
    dataset = []
    possible_symbols = []
    
    for index, seq in enumerate(seq_list):
        values = np.array(list(seq))
        integer_encoded = label_encoder.transform(values)
        #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        #print('integer encoded sequence shape: %s' %(integer_encoded.shape))
        dataset.append(integer_encoded)
        #possible_symbols = possible_symbols + list(label_encoder.inverse_transform(integer_encoded))
        #print(list(label_encoder.inverse_transform(integer_encoded)))
    #print(set(possible_symbols))
    pickle.dump( np.array(dataset), open( "./data/data.p", "wb" ) )
    pickle.dump( np.array(name_list), open( "./data/names.p", "wb" ) )

def preprocess(raw_txt_path = './data/combineddata.fasta'):
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


#def txt_to_csv(raw_txt_path,  position_ind = None, sample_ind = None):
#     keys = []
#     vals = []
#     sequence = ''
    
#     with open(raw_txt_path) as RRM:
#         lines = RRM.readlines()
#         last = lines[-1]
#         for line_index, line in enumerate(RRM):
#             print(line_index)
#             # filtering by samples, skip if the sample indicator variable is 0 at current sample    
#             if sample_ind:
#                 if not sample_ind[line_index]:
#                     continue
            
#             if '>' in line:
#                 keys.append(line[1:-2]) # not a line, this is a line name

#             else:
#                 sequence += line.strip()
#                 next_line = next(RRM)
#                 if line == last:
#                     vals.append(sequence)
#                     break

#                     if '>' in next_line:
#                         if position_ind:
#                             sequence_filtered = []
#                             for i in range(len(position_ind)):
#                                 if position_ind[i]:
#                                     sequence_filtered.append(sequence[i])
#                             sequence = sequence_filtered

#                             vals.append(sequence)
#                             sequence = ''
                
#     if sample_ind: # if position + sample filtered, write fasta version for Seq2Vec
#         write_fasta(vals, keys, fasta_name = './data/RRM_55_sample_position_filtered.fasta')
#         print(len(vals), ' samples made it')

#     print(len(keys), len(vals), vals[0])
#     df = pd.DataFrame({'keys': keys, 'vals': vals})
    
#     # write df if it dosen't fit in RAM 
#     return df
