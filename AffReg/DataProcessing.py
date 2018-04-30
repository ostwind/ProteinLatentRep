import argparse
import numpy as np
from Bio import SeqIO
import pandas as pd
import pickle

def create_parser(args_in=None):
    parser = argparse.ArgumentParser(description="Affinity Regression")
    parser.add_argument('--profiles', type=str, default="../dropbox_files/Fullset_z_scores_setAB.txt", help="profiles")
    parser.add_argument('--pearson_file', type=str, default='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt', help="training set")
    parser.add_argument('--emb_file', type=str, default="hiddens.csv", help="low dimensional embeddings")
    # pytorch arguments 
    args = parser.parse_args()
    return args

def GetLabeledAndCombine(embedding, name_col_str):
    # reprocess names so that *||RNCMPT00232_RRM__0 becomes RNCMPT00232
    # also just filters for labeled data
    embedding.rename(columns={embedding.columns[0]:"name"},inplace=True)
    embedding.columns = ["{0}_le_col".format(x) for x in embedding.columns]

    # lost 13+1 RNACompete experiments here
    embedding = embedding[embedding[name_col_str].str.contains("\|\|")]
    embedding.loc[:, name_col_str] = [x.split("||")[1] for x in embedding[name_col_str]]
    embedding.loc[:,name_col_str] = [x.split("_")[0] for x in embedding[name_col_str]]
    
    embedding = embedding.groupby(name_col_str, as_index=False, axis=0).mean() #? 
    
    return embedding
    
def filter_embs(Y, protnames, embedding, args):
    ''' merges Y's z-scores with embedding file according to protein names, then split.
        prots_final: list of protein names in the order of Y and embedding
    '''
    Y_df = pd.DataFrame(Y)
    Y_df.loc[:, "name_le_col"] = protnames
    total_df = Y_df.merge(embedding, left_on="name_le_col", right_on="name_le_col", how="inner")
    path_295 = '../dropbox_files/Full_proteinlist_RRMprotein_top100pearson.txt'
    
    
    with open(args.pearson_file) as f:
        _ = f.readline()  
        _ = f.readline()
        train = f.readline()
        train2 = f.readline()
        _ = f.readline()
        test = f.readline()

    # with open(path_295) as f: #args.pearson_file) as f:
    #     _ = f.readline()
    #     data = f.readline()

    RRM_prots = '|'.join( train.split(' ') + train2.split(' ') + test.split(' '))# + data.split(' ') )
    #RRM_prots = '|'.join( data.split(' '))# + train2.split(' ') + test.split(' ') )
    
    #print(RRM_prots)
    #!!!
    total_df = total_df[ total_df.name_le_col.str.contains(RRM_prots) ]
    print(total_df.shape)

    #exit()

    Y_final, embs_final = total_df[Y_df.columns], total_df[embedding.columns] 
    prots_final = Y_final.name_le_col
    
    Y_final = Y_final.loc[:, Y_final.columns != 'name_le_col']
    embs_final = embs_final.loc[:, embs_final.columns != 'name_le_col']
    
    return Y_final.as_matrix(), embs_final.as_matrix(), prots_final

def SVD(Y_train, low_rank_dim = 80):
    #D = pickle.load(open( "D.p", "rb" ))
    # np.matmul(Y_train, D)
    U, S, V = np.linalg.svd( Y_train , full_matrices=False)
    #print( 'percent variance explained: ', np.sum( S[:low_rank_dim] )/ np.sum(S) )
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]
    return U #X_train, X_test

def LoadData():
    args = create_parser()
    pearson_file, emb_file, profiles = args.pearson_file, args.emb_file, args.profiles

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    #Y = normalize(Y, axis = 0) #normalize each PBM experiment (row) following supple. sec 1.1
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])    

    if 'kmers.csv' in emb_file:                                         
        embedding = pd.read_csv(emb_file)
    else:
        embedding = pd.read_csv(emb_file, sep='\t')
    
    embedding = GetLabeledAndCombine(embedding, "name_le_col")
    Y, X, prots = filter_embs(Y, ynames, embedding, args)
    
    #Y_test, X_test, testprots_final = filter_embs(Y_test, testprots, embedding)
    #print(Y_train.shape, X_train.shape)
    
    return X, Y, prots 