import argparse
import numpy as np
from Bio import SeqIO
import pandas as pd

def create_parser(args_in=None):
    parser = argparse.ArgumentParser(description="Affinity Regression")
    parser.add_argument('--profiles', type=str, default="../dropbox_files/Fullset_z_scores_setAB.txt", help="profiles")
    parser.add_argument('--pearson_file', type=str, default='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt', help="training set")
    parser.add_argument('--emb_file', type=str, default="hiddens.csv", help="low dimensional embeddings")
    # pytorch arguments 
    args = parser.parse_args()
    return args

def original_script_dataset_processing(datnames,Y, arg1, pearson_file):
    """
    Data processing script from Alex's Affinity regression files
    Gets binding preference expressions for protein names, splits into train
    and test sets based on predefined cuts.
    datnames: Protein names to filter for
    Y: Binding preference matrix
    """
    numset = arg1
    #print( "trainset:", pearson_file, numset)
    tobj = open(pearson_file, 'r')
    tlines = tobj.readlines()
    for i, tline in enumerate(tlines):
    
        if "###Set" in tline: #and tline.strip().split()[-1] == numset:
            c = [tlines[i+2].strip().split(), tlines[i+3].strip().split()]
            trainprots = np.array(c).T
            
            d = [tlines[i+5].strip().split(), tlines[i+6].strip().split()]
            testprots = np.array(d).T
            
            if len(np.intersect1d(trainprots[:,0], datnames)) !=0:
            #  col0: 'RNCMPT00676' vs col1: 'T129595'
                trainprots, testprots = trainprots[:,0], testprots[:,0]
            
            else: 
                trainprots, testprots = trainprots[:,1], testprots[:,1]
            
            indexestrain, indexestest = [], []
            for i, ina in enumerate(datnames):
                if ina in trainprots:
                    indexestrain.append(i)
                elif ina in testprots:
                    indexestest.append(i)
            
            # label, data train/test split
            indexestrain, indexestest = np.array(indexestrain), np.array(indexestest)            
            Ytrain, Ytest = Y[indexestrain], Y[indexestest]
            trainprots, testprots = datnames[indexestrain], datnames[indexestest]
    return Ytrain, Ytest, trainprots, testprots 

def get_labeled(input_df, name_col_str):
    # reprocess names so that *||RNCMPT00232_RRM__0 becomes RNCMPT00232
    # also just filters for labeled data
    filtered_df = input_df[input_df[name_col_str].str.contains("\|\|")]
    filtered_df.loc[:, name_col_str] = [x.split("||")[1] for x in filtered_df[name_col_str]]
    filtered_df.loc[:,name_col_str] = [x.split("_")[0] for x in filtered_df[name_col_str]]
    return filtered_df
    
def average_overlapping_embs(emb_df, name_col):
    out_df = emb_df.groupby(name_col, as_index=False, axis=0).mean() #? 
    return out_df

def LowRank_OrthoMat(matrix, low_rank_dim, LeftOrtho):
    ''' applies SVD to matrix, returns low rank approximation of M, 
        and Left (VS^-1) or Right (S^-1V)^T orthogonal matrix (for reconstructing original W) 
    '''
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]

    matrix_low_rank = np.matmul( np.matmul(U, S) , V)
    if LeftOrtho:
        matrix_LeftMat_reconstruct = np.matmul( V.T, np.linalg.inv(S) )     
        return matrix_low_rank, matrix_LeftMat_reconstruct, U
    else: 
        #print(matrix.shape, U.shape, S.shape, V.shape)
        matrix_RightMat_reconstruct = np.matmul( np.linalg.inv(S), V ).T
        #print(matrix_low_rank.shape, matrix_RightMat_reconstruct.shape)
        return matrix_low_rank, matrix_RightMat_reconstruct

def filter_embs(Y, protnames, le_df):
    ''' merges Y's z-scores with embedding file according to protein names, then split.
        prots_final: list of protein names in the order of Y and embedding
    '''
    Y_df = pd.DataFrame(Y)
    Y_df.loc[:, "name_le_col"] = protnames
    
    total_df = Y_df.merge(le_df, left_on="name_le_col", right_on="name_le_col", how="inner")
  
    Y_final, embs_final = total_df[Y_df.columns], total_df[le_df.columns] 
    prots_final = Y_final.name_le_col
    
    Y_final = Y_final.loc[:, Y_final.columns != 'name_le_col']
    embs_final = embs_final.loc[:, embs_final.columns != 'name_le_col']
    return Y_final.as_matrix(), embs_final.as_matrix(), prots_final

