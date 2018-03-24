from DataProcessing import * 
from scipy.stats import pearsonr 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, average_precision_score
from sklearn.model_selection import KFold 
import pickle        
from numpy import trapz
import matplotlib.pyplot as plt 

def plot(recall, precision, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    plt.show()

def AUC(y_pred, y_test):
    aucs, avg_recall = 0, 0
    num_one_percent_most_intense_probes = int(len(y_test[0])/100)
    for pred, truth in zip(y_pred, y_test):
        # select top 1% of probe intensities in pred
        pred_brightest_indices = np.argsort(pred)#[-100:]
        test_brightest_indices = np.argsort(truth)[-num_one_percent_most_intense_probes:]

        # compute AUC  
        test_brightest = truth[test_brightest_indices]
        pred_brightest = pred[ pred_brightest_indices ]

        test_one_zero = [ x in test_brightest_indices for x in pred_brightest_indices ]
        test_one_zero = np.array(test_one_zero)

        #precision, recall, thresholds = precision_recall_curve(test_one_zero, pred_brightest, pos_label =1) 
        fpr, tpr, thresholds = roc_curve(test_one_zero, pred_brightest, pos_label=1)
        
        #avg_recall += recall_score(y_true = test_one_zero,
        #y_pred = np.ones( len(test_one_zero) ),average = 'binary', pos_label = 1)
        aucs +=  auc(fpr, tpr, )
        avg_recall += average_precision_score(test_one_zero, pred_brightest, average = 'samples') 
        
    return aucs, avg_recall, y_pred.shape[0] 

def ReconstructKNN(yyt_pred,  y_test,  y_train):
    preds = []
    for test_sample_sim in yyt_pred:
        most_similar_indices = np.argsort(test_sample_sim)[-5:] # indices of highest similarities  
        most_similar_profiles = y_train[most_similar_indices, :]

        # compute weights for weighted sum of nearest profiles to held-out sample
        similarities = test_sample_sim[most_similar_indices]
        weights = similarities/np.sum(similarities)

        y_pred = np.multiply( most_similar_profiles, weights[:, np.newaxis] )
        y_pred = np.sum(y_pred, axis = 0)
        
        preds.append(y_pred)
        
    auc = AUC( np.array(preds), y_test)
    return auc 

def Inference(model, test_data, similarity_shape):
    Y_test_squared = model.predict(test_data) 
    Y_test_squared =  Y_test_squared.reshape(similarity_shape, order = 'F') # 24 X 213 
    # |test_samples (Kron) train_samples | -> |test_samples| X |train_samples| where columns are  un-concatenated
    
    return Y_test_squared

def TrainRidge(X, y, lamb):
    # X already zero-centered, unit-var
    clf = Ridge(alpha = lamb, copy_X=True,
    fit_intercept=True, normalize = True, random_state=None,  tol=0.0001)
    clf.fit(X, y) 
    return clf

def kfold_validation(X, y, lamb ):
    print('kfold ', X.shape, y.shape)
    kf =KFold(n_splits = 10)
    kfoldAUC, sample_num, avg_precs = 0, 0, 0
    for index, (train_index, test_index) in enumerate(kf.split(X)) :
        #print("TRAIN:", train_index, "TEST:", test_index)
        embs_train, embs_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        X_train, X_test = preprocess(Y_train, embs_train, embs_test)

        yyt =  np.matmul(Y_train, Y_train.T)
        yyt_vec = yyt.flatten('F') # flatten YYT into a long concatenation of its cols
        #test_samples, num_training_samp = embs_.shape[0], yyt.shape[1]
    
        model = TrainRidge( X_train, yyt_vec, lamb)
        YYT_TestPred = Inference(
            model, X_test, similarity_shape = (embs_test.shape[0], embs_train.shape[0]))
        avg_auc, avg_precision, fold_sample_num = ReconstructKNN(
            yyt_pred = YYT_TestPred, #yyt_train = yyt_train,
            y_test = Y_test,  y_train = Y_train )
        kfoldAUC += avg_auc
        sample_num += fold_sample_num
        avg_precs += avg_precision        

        print(index)

    return kfoldAUC  / sample_num, avg_precs / sample_num 

def preprocess(Y_train, embs_train, embs_test):
    low_rank_dim = 80 # ideally t = 100%
    U, S, V = np.linalg.svd( Y_train , full_matrices=False)
    #print( 'percent variance explained: ', np.sum( S[:low_rank_dim] )/ np.sum(S) )
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]
    
    #print(embs_train.shape, U.shape)
    X_train, X_test = np.kron( embs_train, U), np.kron(embs_test, U) # (64, 201) (201, 80)
    
    #test_samples, num_training_samp = embs_test.shape[0], yyt.shape[1]
    return X_train, X_test#, yyt_vec

def ParameterSearch(X_train, X_test, Y_train, Y_test,
     yyt_vec, similarity_shape ):
    #solve: min_Wt || vec(Y^T Y) - ( Kron(P, Ut) ) vec(Wt)_2^2 || + lambda || vec(Wt) ||_2
    for lamb in list(np.arange(0, 10 , 0.5)):
        model = TrainRidge( X_train, yyt_vec, lamb)
        YYT_TestPred = Inference(
            model, X_test, similarity_shape = similarity_shape)
        avg_auc, avg_prec, num_samples = ReconstructKNN(
            yyt_pred = YYT_TestPred, #yyt_train = yyt_train,
            y_test = Y_test,  y_train = Y_train )
        
        print( lamb, avg_auc/num_samples, avg_prec/num_samples )
        #print(lamb, '%.4e'%avg_pcc, '%.4e' %avg_abs_error)

def main():
    args = create_parser()
    pearson_file, emb_file, profiles = args.pearson_file, args.emb_file, args.profiles

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    #Y = normalize(Y, axis = 0) #normalize each PBM experiment (row) following supple. sec 1.1
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])    
    
    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(
        ynames, Y, arg1="0", pearson_file=pearson_file)
    
    if 'kmers.csv' in emb_file:                                         
        embedding = pd.read_csv(emb_file)#, sep='\t')
    else:
        embedding = pd.read_csv(emb_file, sep='\t')
    
    embedding.rename(columns={embedding.columns[0]:"name"},inplace=True)
    embedding.columns = ["{0}_le_col".format(x) for x in embedding.columns]
    embedding = get_labeled(embedding, "name_le_col")
    embedding = average_overlapping_embs(embedding, "name_le_col")
    
    Y_train, embs_train, trainprots_final = filter_embs(Y_train, trainprots, embedding)
    Y_test, embs_test, testprots_final = filter_embs(Y_test, testprots, embedding)
    
    #print(Y_train.shape, Y_test.shape, embs_train.shape, embs_test.shape)

    if 1 == 1:
        kfold_auc, avg_prec = kfold_validation( np.vstack((embs_train, embs_test)), np.vstack((Y_train, Y_test)),
        lamb = 1 )
        print('ten-fold auc', kfold_auc, avg_prec) # test 128_good_model for reg = 0 and 0.5

    # D = pickle.load(open( "D.p", "rb" ))
    # np.matmul(Y_train, D)

    else:
        yyt =  np.matmul(Y_train, Y_train.T)
        yyt_vec = yyt.flatten('F') # flatten YYT into a long concatenation of its col
        X_train, X_test = preprocess(Y_train, embs_train, embs_test)
        ParameterSearch(X_train, X_test, Y_train, Y_test,
        yyt_vec, (embs_test.shape[0], embs_train.shape[0])  )

if __name__ == '__main__':
    main()

def AbsoluteError(y_pred, y_test):
    test_relative_error = 0 
    for pred, truth in zip(y_pred, y_test):
        top_binding_kmer_indices = np.argsort(truth)[-10:]
        test_relative_error += np.linalg.norm( pred[top_binding_kmer_indices] - truth[top_binding_kmer_indices] )   
        
    return test_relative_error/pred.shape[0] #avg RelativeErr across test set

def AvgPearson(y_pred, y_test):
    avgpearson = 0
    for pred, truth in zip(y_pred, y_test):
        top_binding_kmer_indices = np.argsort(truth)[-10:] 
        pred_norm, truth_norm  =  pred[top_binding_kmer_indices], truth[top_binding_kmer_indices] 
    
        p_val, corr_coeff = pearsonr( pred_norm , truth_norm )
        avgpearson += corr_coeff
    
    return avgpearson/y_pred.shape[0]


