import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO
from similarity_regression import SimilarityRegression
from data_loading import LowDimData
from train import training_loop
import pandas as pd
import argparse

def create_parser(args_in=None):
    parser = argparse.ArgumentParser(description="Similarity Regression")
    parser.add_argument('--dtype', type=str, default='--z', help="dtype for loading original data")
    parser.add_argument('--profiles', type=str, default="../dropbox_files/Fullset_z_scores_setAB.txt", help="profiles")
    parser.add_argument('--probefeatures', type=str, default="ones", help="profiles")
    parser.add_argument('--proteinfeatures', type=str, default="Proteinsequence_latentspacevectors.txt", help="profiles")
    parser.add_argument('--file_2', type=str, default='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt', help="training set")
    parser.add_argument('--emb_file', type=str, default="hiddens.csv", help="low dimensional embeddings")
    # pytorch arguments 
    parser.add_argument('--optim', type=str, default="SGD", help="optimizer to use for learning weights (defaults to SGD)")
    parser.add_argument('--loss', type=str, default="SGD", help="Loss to use for training")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optim")
    parser.add_argument('--eval_every', type=int, default=5000, help="print auc on dev set every iterations (default: 5000)")
    parser.add_argument('--print_every', type=int, default=100, help="print auc on dev set every iterations (default: 5000)")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size (default 16)")
    parser.add_argument('--num_epochs', type=int, default=800000, help="batch size (default 16)")
    parser.add_argument('--emb_dim', type=int, default=250, help="batch size (default 16)")
    parser.add_argument('--rna_dim', type=int, default=203, help="batch size (default 16)")
    parser.add_argument('--use_cuda', action='store_true', help="batch size (default 16)")
    
    if args_in is not None:
        args = parser.parse_args(args_in.split(" "))
    else:
        args = parser.parse_args()

    return args

def low_rank_approx(SVD=None, A=None, r=None):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not r:
        r = A.shape[0]

    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=True)
    u, s, v = SVD
    print(s, np.sum(s)/np.sum(s[:r]), ' variance explained')
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u[i], v[i])
    return Ar

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
        #print(tline)
    
        if "###Set" in tline: #and tline.strip().split()[-1] == numset:
            
            #a = tlines[i+2]
            #b = tlines[i+2].strip().split()
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
            
            #print ('train shape = %s, test shape = %s, %s '  %(np.shape(Ytrain), np.shape(Ytest), np.shape(Y)) ) 
            # train: 213, test: 24, total: 407, unused: 170 

    return Ytrain, Ytest, trainprots, testprots 

# preprocessing on embedding df
def split_labeled_protein_names(input_df, name_col_str):
    # reprocess names so that *||RNCMPT00232_RRM__0 becomes RNCMPT00232
    # also just filters for labeled data
    filtered_df = input_df[input_df[name_col_str].str.contains("\|\|")]
    filtered_df[name_col_str] = [x.split("||")[1] for x in filtered_df[name_col_str]]
    filtered_df[name_col_str] = [x.split("_")[0] for x in filtered_df[name_col_str]]
    return filtered_df
    
def average_overlapping_embs(emb_df, name_col):
    print(emb_df.shape)
    out_df = emb_df.groupby(name_col, as_index=False, axis=0).mean()
    print(out_df.shape)
    return out_df

def filter_embs(Y, protnames, le_df):
    Y_df = pd.DataFrame(Y)
    Y_df["name_le_col"] = protnames
    
    total_df = Y_df.merge(le_df, left_on="name_le_col", right_on="name_le_col", how="inner")
    Y_final, embs_final = total_df[Y_df.columns], total_df[le_df.columns] 
    prots_final = Y_final.name_le_col
    #embs_final.fillna(value=0, inplace=True)
    
    Y_final = Y_final.loc[:, Y_final.columns != 'name_le_col']
    embs_final = embs_final.loc[:, embs_final.columns != 'name_le_col']
    
    return Y_final.as_matrix(), embs_final.as_matrix(), prots_final
        
def main():
    args = create_parser()

    # naming variables for convenience with legacy code
    dtype, profiles, probefeatures, proteinfeatures = args.dtype, args.profiles, args.probefeatures, args.proteinfeatures

    #### get feature matrices and names
    pearson_file, emb_file = args.file_2, args.emb_file


    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])    
    #sevenmers = np.genfromtxt(profiles, skip_header=1)[:, 0]
    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(
        ynames,Y, arg1="0", pearson_file=pearson_file)
                                                                               
    learned_embs_df = pd.read_csv(emb_file, sep='\t')
    learned_embs_df.rename(columns={learned_embs_df.columns[0]:"name"},inplace=True)
    learned_embs_df.columns = ["{0}_le_col".format(x) for x in learned_embs_df.columns]

    learned_renamed = split_labeled_protein_names(learned_embs_df, "name_le_col")
    learned_renamed = average_overlapping_embs(learned_renamed, "name_le_col")

    Y_train_final, embs_train, trainprots_final = filter_embs(Y_train, trainprots, learned_renamed)
    Y_test_final, embs_test, testprots_final = filter_embs(Y_test, testprots, learned_renamed)

    #Y_train_final = low_rank_approx(SVD=None, A=Y_train_final, r=False)
    #yyt = Y_train_final

    yyt = np.dot(Y_train_final, Y_train_final.T)
    
    # project Y_test_final onto Y_train_final to approx proj onto singular vectors
    yyt_dev = np.dot(Y_test_final, Y_train_final.T)
    #print(yyt_dev.shape)
    #yyt_dev = low_rank_approx(SVD=None, A = Y_test_final, r = 24 ) 
    #print(yyt_dev.shape)

    print("embs shape", embs_train.shape)
    learned_embs =torch.FloatTensor(embs_train) # torch.randn((213,10))
    # replacing YYT on LHS with transposed embeddings 
    poss_matches = torch.FloatTensor(embs_train.T) 

    known_matches = torch.FloatTensor(yyt) 
    
    learned_embs_dev =torch.FloatTensor(embs_test) # torch.randn((213,10))
    poss_matches_dev = torch.FloatTensor(yyt_dev) 
    known_matches_dev = torch.FloatTensor(yyt_dev) 
    
    (args.rna_dim, args.emb_dim) = embs_train.shape
    test_model = SimilarityRegression(emb_dim=args.emb_dim, rna_dim=args.rna_dim)
    #for x in test_model.parameters():
    #    x.data = x.data.normal_(0.0, 0.5)
    
    if args.use_cuda:
        learned_embs = learned_embs.cuda()
        poss_matches = poss_matches.cuda()
        known_matches = known_matches.cuda()
        learned_embs_dev = learned_embs_dev.cuda()
        poss_matches_dev = poss_matches_dev.cuda()
        known_matches_dev = known_matches_dev.cuda()
        test_model.cuda()
     
    optimizer = optim.Adam(test_model.parameters(), lr=args.lr) #, betas=(0.5, 0.999)) # ?
    
    #test_model.init_weights()
    
    input_data = DataLoader(LowDimData(learned_embs, 
                                       known_matches), 
                            batch_size= args.batch_size)
    dev_input_data = DataLoader(LowDimData(learned_embs_dev, 
                                           known_matches_dev), 
                                batch_size=args.batch_size)
    training_loop(args.batch_size, 
                  args.num_epochs, 
                  test_model, 
                  optimizer, 
                  input_data,  ##
                  poss_matches, 
                  dev_input_data, 
                  embed_file = args.emb_file,
                  print_every=args.print_every, 
                  eval_every=args.eval_every)    

if __name__ == '__main__':
    main()

def main_test():
    # dummy data
    print("Testing functionality of Similarity Regression")
    learned_embs = torch.randn((777,10))
    poss_matches = torch.randn((2555, 30))
    known_matches = torch.randn((777,30))

    test_model = SimilarityRegression(emb_dim=10, rna_dim=2555)
    optimizer = SGD(test_model.parameters(), lr = 0.001)
    test_model.init_weights()
    # input_data = (learned_embs, known_matches)
    batch_size = 5
    num_epochs = 5
    input_data = DataLoader(LowDimData(learned_embs, known_matches), batch_size=batch_size)
    training_loop(batch_size, num_epochs, test_model, optimizer, input_data, poss_matches)
