import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO
from affinity_regression import AfinityRegression
from data_loading import LowDimData
from train import training_loop
import pandas as pd
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Affinity Regression")
    parser.add_argument('--dtype', type=str, default='--z', help="dtype for loading original data")
    parser.add_argument('--profiles', type=str, default="../dropbox_files/Fullset_z_scores_setAB.txt", help="profiles")
    parser.add_argument('--probefeatures', type=str, default="ones", help="profiles")
    parser.add_argument('--proteinfeatures', type=str, default="Proteinsequence_latentspacevectors.txt", help="profiles")
    parser.add_argument('--file_2', type=str, default='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt', help="training set")
    parser.add_argument('--emb_file', type=str, default="hiddens.csv", help="low dimensional embeddings")
    return parser


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




def filter_embs(Y, protnames, le_df):
    Y_df = pd.DataFrame(Y)
    Y_df["name_le_col"] = protnames
    total_df = Y_df.join(le_df, on="name_le_col", rsuffix="le_col", how="left")
    print("TOTAL_DF SHAPE: ", total_df.shape)
    Y_final = total_df[Y_df.columns]
    embs_final = total_df[le_df.columns]
    prots_final = Y_final.name_le_col
    embs_final.fillna(value=0, inplace=True)
    print(embs_final.columns)
    return Y_final.as_matrix()[:,:-1], embs_final.as_matrix()[:,:-1], prots_final




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


    
def main_real():
    parser = create_parser()
    args = parser.parse_args()

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
    Y_train_final, embs_train, trainprots_final = filter_embs(Y_train, trainprots, learned_embs_df)
    Y_test_final, embs_test, testprots_final = filter_embs(Y_test, testprots, learned_embs_df)



    yyt = np.dot(Y_train_final, Y_train_final.T)

    print("embs shape", embs_train.shape)
    learned_embs =torch.FloatTensor(embs_train) # torch.randn((213,10))
    poss_matches = torch.FloatTensor(yyt) 
    known_matches = torch.FloatTensor(yyt) 

    test_model = AfinityRegression(emb_dim=10, rna_dim=213)
    optimizer = SGD(test_model.parameters(), lr = 0.001)
    test_model.init_weights()
    # input_data = (learned_embs, known_matches)
    batch_size = 5
    num_epochs = 100000
    input_data = DataLoader(LowDimData(learned_embs, known_matches), batch_size=batch_size)
    training_loop(batch_size, num_epochs, test_model, optimizer, input_data, poss_matches)
    
if __name__ == '__main__':
    main_real()