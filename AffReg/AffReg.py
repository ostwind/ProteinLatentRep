from DataProcessing import * 
import scipy.linalg as sl
from scipy.spatial.distance import cosine    
from scipy.cluster.vq import whiten 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge 
import pickle        

def MSE(pred, label):
    test_set_error = 0 
    for sample_pred, sample_label in zip(pred, label):
        test_set_error += np.abs(pred - label).mean() # MAE of a single sample
    return test_set_error/pred.shape[0] #avg MAE across test set

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

    #print(np.array(preds).shape, yyt_pred.shape, y_train.shape)
    mse = MSE(np.array(y_pred), y_test)
    return mse

def Inference(model, test_data, test_label, similarity_shape):
    Y_test_squared = model.predict(test_data) 
    Y_test_squared = np.array( Y_test_squared ).reshape(similarity_shape, order = 'F') # 24 X 213 
    # |test_samples (Kron) train_samples | -> |test_samples| X |train_samples| where columns are  un-concatenated
    return Y_test_squared

def TrainRidge(X, y, lamb):
    # X already zero-centered, unit-var
    clf = Ridge(alpha = lamb, copy_X=False, fit_intercept=False, random_state=None,  tol=0.0001)
    clf.fit(X, y) 
    return clf

def main():
    args = create_parser()
    pearson_file, emb_file, profiles = args.pearson_file, args.emb_file, args.profiles

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    Y = normalize(Y, axis = 1) #normalize each PBM experiment (row) as per supple. sec 1.1
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
    
    embs_train = StandardScaler(with_std = False).fit_transform(embs_train )
    embs_test  =  StandardScaler(with_std = False).fit_transform(embs_test )

    if 'kmers.csv' in emb_file:
        embs_train, embs_test = normalize(embs_train, axis = 1), normalize(embs_test, axis= 1) 
    else:
        embs_train, embs_test = normalize(embs_train, axis = 0), normalize(embs_test, axis= 0) 


    print(Y_train.shape, Y_test.shape, embs_train.shape, embs_test.shape)

    low_rank_dim = 80 # ideally t = 100%
    
    D = pickle.load(open( "D.p", "rb" ))

    U, S, V = np.linalg.svd( Y_train , full_matrices=False)
    print( 'percent variance explained: ', np.sum( S[:low_rank_dim] )/ np.sum(S) )
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]
    
    X = np.kron( embs_train, U)
    X_test = np.kron(embs_test, U) 
    
    y =  np.matmul(Y_train, Y_train.T)
    yyt_train = np.matmul(Y_train,Y_train.T)

    test_samples, num_training_samp = embs_test.shape[0], y.shape[1]
    y = y.flatten('F') # flatten YYT into a long concatenation of its cols
        
    #solve: min_Wt || vec(Y^T Y) - ( Kron(P, Ut) ) vec(Wt)_2^2 || + lambda || vec(Wt) ||_2
    for lamb in list(np.arange(0, 10 , 0.5)):
        model = TrainRidge( X, y, lamb)
        YYT_TestPred = Inference(
            model, X_test, Y_test,  similarity_shape = (test_samples, num_training_samp))
        mse = ReconstructKNN(
            yyt_pred = YYT_TestPred, #yyt_train = yyt_train,
            y_test = Y_test,  y_train = Y_train )
        print(lamb, mse)


if __name__ == '__main__':
    main()

def ReconstructY(YYT_test, y_train):
    ''' detailed in ar_reconstruction.m in supplementary code 
        reconstructs Y from predicted YY^T, reconstructs Y_test from linear comb of Y_train span 
    '''
    YYT_test = YYT_test.T
    #y_train =  y_train.T # (num_samples, num DNA) -> (num DNA, num samples)

    A = np.matmul( y_train.T, YYT_test)
    O = sl.orth(y_train) # 213 X 213 as many basis vectors as training examples 
    
    #print( O.shape, A.shape, y_train.shape )
    c, _,_,_ = np.linalg.lstsq(O, y_train) # np.linalg.solve(O, y_train) #
    
    #print(c.shape)
    #(213, 16382) (16382, 24) (213, 16382)

    ct, _,_,_ =  np.matmul( np.linalg.inv(c.T), A ) # np.linalg.lstsq(c.T, A) #
    
    print(y_train.shape, YYT_test.shape, A.shape, O.shape, c.shape,ct.shape, np.matmul(O, ct).shape )
    #(213, 16382) (213, 24) (16382, 24) (213, 213) (213, 16382) (213, 24) (213, 24)
    return np.matmul(O, ct)
