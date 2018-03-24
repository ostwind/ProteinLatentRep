''' this file's dir should be sibling to data dir

    transform raw text into csv
    eliminating non informative positions
    integer encode (optionally one-hot encode) and pickle
    pytorch loader (in loader.py) simply unpacks then yields samples
'''

import sys
import os
parent_dir = os.path.abspath(__file__ + "/../../")
sys.path.append(parent_dir)

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
    vals, keys = [], []
    with open(raw_txt_path) as RRM:
        for line_index, line in enumerate(RRM):
            non_empty_strings = [s for s in line.split(' ') if s]
            protein_name, protein = non_empty_strings[0], non_empty_strings[-1]
            #if not position_ind: # nothing filtered, do text tuncating
            # remove '>' from name, '/' -> '_', delete '\n' in protein seq
            protein_name, protein = protein_name[1:].replace("/","@"), protein[:-2] 
            
            # filtering by samples, skip if the sample indicator variable is 0 at current sample    
            if sample_ind: # filtered, last call before pickling
                if '||' not in protein_name: # if its unlabeled data AND was filtered as sample: 
                    if not sample_ind[line_index]:
                        continue
            sequence_filtered = protein
            if position_ind:
                # filtering by positions
                sequence_filtered = []
                for i in range(len(position_ind)):
                    if position_ind[i]: # 1/0 indicator bit
                        sequence_filtered.append( protein[i] )
                # if  line_index ==2001:
                #     #print( "".join(sequence_filtered) )
                #     print( keys[2000] )
                #     print( keys[2] )
                #     print( vals[2000])
                #     print( vals[2])
                #     print('_____________________________')
            keys.append(protein_name)
            vals.append( "".join(sequence_filtered) )

    if sample_ind: # if position + sample filtered, write fasta version for Seq2Vec
        write_fasta(vals, keys, fasta_name = './data/RRM55_with_labeled.fasta')
        #print(len(vals), ' samples made it')
    
    df = pd.DataFrame({'keys': keys, 'vals': vals})
    return df

def _filter_positions(df, threshold = 0.01, plot=False):
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

def _filter_samples(df, threshold = 0.4, plot = False):
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

def one_hot_pickle(df2, path):
    seq_list, name_list = df2['vals'].tolist(), df2['keys'].tolist()
 
    label_encoder = LabelEncoder()
    
    symbols = ['SOS'] + list(set( ''.join(seq_list ) ) )
    label_encoder.fit( symbols )

    integers = label_encoder.transform(symbols)
    pickle.dump( dict(zip(integers, symbols)), open( "./data/integer_to_symbol_dictionary.p", "wb") )

    print(dict(zip(integers, symbols)) )
    #print('SOS token int encoded as: ', label_encoder.transform(['SOS']))
    
    dataset, indices = [], []
    
    for index, seq in enumerate(seq_list):
        values = np.array(list(seq))
        integer_encoded = label_encoder.transform(values)
        #print('integer encoded sequence shape: %s' %(integer_encoded.shape))
        dataset.append(integer_encoded)
        #print(list(label_encoder.inverse_transform(integer_encoded)))
        indices.append(index)

    pickle.dump( np.array(indices), open( path + "indices.p", "wb" ) )
    pickle.dump(  np.array(dataset), open( path + "data.p", "wb" ) )
    pickle.dump( np.array(name_list), open( path + "names.p", "wb" ) )


def preprocess(raw_txt_path = './data/combineddata.fasta'):
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)
    
    df = txt_to_csv(raw_txt_path, ) # first convert to csv
    #filter empty positions then re-write csv to informative_csv_path
    
    position_ind = _filter_positions(df)
    df1 = txt_to_csv(raw_txt_path,  position_ind = position_ind)
    
    sample_ind = _filter_samples(df1)
    df2 = txt_to_csv(raw_txt_path, position_ind = position_ind,
    sample_ind = sample_ind,)
    df2 = df2.drop_duplicates(subset='vals', keep="last") # 68771 -> 52052
    print(df2.shape, ' samples made it')    

    ################ aligned/unaligned/delimited ###################3
    # original_len = len(df2['vals'][0])
    
    # #df2['vals'] = df2['vals'].str.replace('\-\-+', '-') + ''.join( ['-' for i in range(78)] )
    # '''
    # VFLGGV-----EA----TF--------W-------------------G-YLVFELEKSVRSLL--C------------
    # VFLGGV-EA-TF-W-G-YLVFELEKSVRSLL-C---------------------------------------------
    # '''
    
    # df2['vals'] = df2['vals'].str.replace('\-', '') + ''.join( ['-' for i in range(original_len)] )
    # '''
    # VFLGGV-----EA----TF--------W-------------------G-YLVFELEKSVRSLL--C------------
    # VFLGGVEATFWGYLVFELEKSVRSLLC---------------------------------------------------
    # '''
    
    # df2['vals'] = df2['vals'].apply(lambda x: x[:78])
    
    #################################################
    #print(df2['vals'])
    one_hot_pickle(df2, path = './data/aligned/')
    

    count = 0
    thefile = open('test.txt', 'w')
    for item in df2['keys'].tolist():
        if '||' in item:
            count += 1
    print(count)
    # thefile.write("%s\n" % item)

    #sorted_proteins = df2['vals'].tolist()
    #df
    #sorted_proteins.sort()

    # from itertools import groupby
    # freq = [len(list(group)) for key, group in groupby(sorted_proteins)] 
    
    # for item in sorted_proteins:
    #     print(item)
    #     print('\n')
    #print(freq)
if __name__ == '__main__':
      preprocess()

#def _form_proteins(df):
#     ''' takes in all labeled RRMs, bin them into their proteins, pickle file with data and name
#     '''
#     rrm_names, rrms = df['keys'].tolist(), df['vals'].tolist()
#     protein_dictionary= dict()
#     #protein_dictionary = {k: [[], [], [], [], []] for k in range(800)}

#     for index, (rrm_name, rrm) in enumerate( zip(rrm_names, rrms) ):
#         protein_name, rrm_position = rrm_name.split('_RRM__')[0], int(rrm_name.split('_RRM__')[-1]) 

#         if protein_name not in protein_dictionary.keys():
#             protein_dictionary[protein_name] = ['','','','']
#             protein_dictionary[protein_name][rrm_position] = rrm             
#         else:
#             protein_dictionary[protein_name][rrm_position] = rrm             
