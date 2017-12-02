import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO
from affinity_regression import AfinityRegression
from data_loading import LowDimData
from train import training_loop
import pandas as pd
import argparse

def create_parser(args_in=None):
    parser = argparse.ArgumentParser(description="Affinity Regression")
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

    
    
    if args_in is not None:
        args = parser.parse_args(args_in.split(" "))
    else:
        args = parser.parse_args()

    return args


def original_script_dataset_processing(datnames,Y, arg1="0(,24,48...)", arg2=""):
    numset = arg1
    trainfile = arg2
    
    print( "trainset:", trainfile, numset)
    tobj = open(trainfile, 'r')
    tlines = tobj.readlines()
    for i, tline in enumerate(tlines):
#         print(tline[:6], tline[:6]=="###Set")
#         if tline[:6]=="###Set":
#             print(tline.strip().split()[-1])
        if tline[:6]=="###Set" and tline.strip().split()[-1] == numset:
            print(i)
            trainprots = np.array([tlines[i+2].strip().split(), tlines[i+3].strip().split()]).T
            testprots = np.array([tlines[i+5].strip().split(), tlines[i+6].strip().split()]).T
#             break
#         else:
#             continue

                        ### sort into training and test set
            if len(np.intersect1d(trainprots[:,0], datnames)) !=0:
                trainprots = trainprots[:,0]
                testprots = testprots[:,0]
            else: 
                trainprots = trainprots[:,1]
                testprots = testprots[:,1]
#             print('trainprots: \n')
#             print(trainprots)
#             print('testprots: \n')
#             print(testprots)
            indexestrain = []
            indexestest = []
            for i, ina in enumerate(datnames):
                if ina in trainprots:
                    indexestrain.append(i)
                elif ina in testprots:
                    indexestest.append(i)
            indexestrain = np.array(indexestrain)
            indexestest = np.array(indexestest)
#             print('indexestrain: \n')
#             print(indexestrain)
            Ytrain = Y[indexestrain]
#             Ptrain = P[indexestrain]
            Ytest = Y[indexestest]
#             Ptest = P[indexestest]
            trainprots = datnames[indexestrain]
            testprots = datnames[indexestest]

            print (np.shape(Ytest)) #, np.shape(Ptest))
            print( np.shape(Ytrain)) #, np.shape(Ptrain))
    return Ytrain, Ytest, trainprots, testprots #Ptrain, Ytest, Ptest



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
    Y_final = total_df[Y_df.columns]
    embs_final = total_df[le_df.columns]
    prots_final = Y_final.name_le_col
    #embs_final.fillna(value=0, inplace=True)
    
    Y_final = Y_final.loc[:, Y_final.columns != 'name_le_col']
    embs_final = embs_final.loc[:, embs_final.columns != 'name_le_col']
    
    return Y_final.as_matrix(), embs_final.as_matrix(), prots_final
    


def main():
    # dummy data
    print("Testing functionality of Affinity Regression")
    learned_embs = torch.randn((777,10))
    poss_matches = torch.randn((2555, 30))
    known_matches = torch.randn((777,30))

    test_model = AfinityRegression(emb_dim=10, rna_dim=2555)
    optimizer = SGD(test_model.parameters(), lr = 0.001)
    test_model.init_weights()
    # input_data = (learned_embs, known_matches)
    batch_size = 5
    num_epochs = 5
    input_data = DataLoader(LowDimData(learned_embs, known_matches), batch_size=batch_size)
    training_loop(batch_size, num_epochs, test_model, optimizer, input_data, poss_matches)


    
    
    
def evaluate_predictions():
    return

    
def main_real():
    args = create_parser()
    print(args)

    dtype = args.dtype # '--z' #sys.argv[sys.argv.index("--data")+1]
    profiles = args.profiles # "../dropbox_files/Fullset_z_scores_setAB.txt" #sys.argv[sys.argv.index("--data")+2]
    probefeatures = args.probefeatures # "ones" #sys.argv[sys.argv.index("--data")+3]
    proteinfeatures = args.proteinfeatures # "Proteinsequence_latentspacevectors.txt" # sys.argv[sys.argv.index("--data")+4]
    #### get feature matrices and names
    file_2 = args.file_2 #'../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt'
    emb_file = args.emb_file # "hiddens.csv"

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    sevenmers = np.genfromtxt(profiles, skip_header=1)[:, 0]
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])

    print(Y.shape)


    Y = np.nan_to_num(Y).T
    D = []
    datnames= ynames

    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(ynames,Y, arg1="0", arg2=file_2)
    if emb_file == "hidden.csv":
        learned_embs_df = pd.read_csv(emb_file)
    else:
        learned_embs_df = pd.read_csv(emb_file, sep='\t')
        learned_embs_df.rename(columns={learned_embs_df.columns[0]:"name"},inplace=True)





    learned_embs_df.columns = ["{0}_le_col".format(x) for x in learned_embs_df.columns]

    learned_renamed = split_labeled_protein_names(learned_embs_df, "name_le_col")
    learned_renamed = average_overlapping_embs(learned_renamed, "name_le_col")



    Y_train_final, embs_train, trainprots_final = filter_embs(Y_train, trainprots, learned_renamed)
    Y_test_final, embs_test, testprots_final = filter_embs(Y_test, testprots, learned_renamed)


    yyt = np.dot(Y_train_final, Y_train_final.T)

    print("embs shape", embs_train.shape)
    learned_embs =torch.FloatTensor(embs_train) # torch.randn((213,10))
    poss_matches = torch.FloatTensor(yyt) 
    known_matches = torch.FloatTensor(yyt) 

    test_model = AfinityRegression(emb_dim=250, rna_dim=203)
    #for x in test_model.parameters():
    #    x.data = x.data.normal_(0.0, 0.5)
    
    if args.optim =='adam':
        optimizer = optim.Adam(test_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.SGD(test_model.parameters(), lr =args.lr)
    
    test_model.init_weights()
    # input_data = (learned_embs, known_matches)
    batch_size = 5
    num_epochs = 100000
    input_data = DataLoader(LowDimData(learned_embs, known_matches), batch_size=batch_size)
    training_loop(batch_size, num_epochs, test_model, optimizer, input_data, poss_matches)    
if __name__ == '__main__':
    main_real()