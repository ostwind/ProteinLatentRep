import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO



def original_script_dataset_processing(datnames=ynames, arg1="0(,24,48...)", arg2=file_2):
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
    
    dtype = '--z' #sys.argv[sys.argv.index("--data")+1]
    profiles = "dropbox_files/Fullset_z_scores_setAB.txt" #sys.argv[sys.argv.index("--data")+2]
    probefeatures = "ones" #sys.argv[sys.argv.index("--data")+3]
    proteinfeatures = "Proteinsequence_latentspacevectors.txt" # sys.argv[sys.argv.index("--data")+4]
    #### get feature matrices and names

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    sevenmers = np.genfromtxt(profiles, skip_header=1)[:, 0]
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])

    print(Y.shape)


    Y = np.nan_to_num(Y).T
    D = []
    datnames= ynames

    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(arg1="0")

    yyt = np.dot(Y_train, Y_train.T)
    
    
    learned_embs = torch.randn((213,10))
    poss_matches = torch.FloatTensor(yyt) # torch.randn((2555, 30))
    known_matches = torch.FloatTensor(yyt) # torch.randn((213,30))

    test_model = AfinityRegression(emb_dim=10, rna_dim=213)
    optimizer = SGD(test_model.parameters(), lr = 0.001)
    test_model.init_weights()
    # input_data = (learned_embs, known_matches)
    batch_size = 5
    num_epochs = 100000
    input_data = DataLoader(LowDimData(learned_embs, known_matches), batch_size=batch_size)
    training_loop(batch_size, num_epochs, test_model, optimizer, input_data, poss_matches)
    
if __name__ == '__main__':
    main()