from DataProcessing import * 
import scipy.linalg as sl
from scipy.spatial.distance import cosine    
from scipy.cluster.vq import whiten 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge 
import pickle        

def AbsoluteError(pred, label):
    test_relative_error = 0 
    for p, l in zip(pred, label):
        test_relative_error += np.linalg.norm( p-l )   
        # relative error of a single sample, Euclidean
    return test_relative_error/pred.shape[0] #avg RelativeErr across test set

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
    test_error = AbsoluteError(np.array(y_pred), y_test)
    return test_error

def Inference(model, test_data, similarity_shape):
    Y_test_squared = model.predict(test_data) 
    Y_test_squared = np.array( Y_test_squared ).reshape(similarity_shape, order = 'F') # 24 X 213 
    # |test_samples (Kron) train_samples | -> |test_samples| X |train_samples| where columns are  un-concatenated
    return Y_test_squared

def TrainRidge(X, y, lamb):
    # X already zero-centered, unit-var
    clf = Ridge(alpha = lamb, copy_X=True, fit_intercept=False, random_state=None,  tol=0.0001)
    clf.fit(X, y) 
    return clf

def main():
    args = create_parser()
    pearson_file, emb_file, profiles = args.pearson_file, args.emb_file, args.profiles

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    Y = normalize(Y, axis = 1) #normalize each PBM experiment (row) following supple. sec 1.1
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])    
    
    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(
        ynames, Y, arg1="0", pearson_file=pearson_file)
    
    #for dim in [ 200]:
    #dim = 200
    emb_file = args.emb_file #'%s_good_model.csv' %(args.dim)
    print(emb_file)

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

    # normalizing by sample better than by feature across all representations
    embs_train, embs_test = normalize(embs_train, axis = 1), normalize(embs_test, axis= 1) 
    
    #print(Y_train.shape, Y_test.shape, embs_train.shape, embs_test.shape)

    low_rank_dim = 80 # ideally t = 100%
    
    #D = pickle.load(open( "D.p", "rb" ))
    #np.matmul(Y_train, D)
    U, S, V = np.linalg.svd( Y_train , full_matrices=False)
    #print( 'percent variance explained: ', np.sum( S[:low_rank_dim] )/ np.sum(S) )
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]
    
    X = np.kron( embs_train, U)
    X_test = np.kron(embs_test, U) 
    
    yyt =  np.matmul(Y_train, Y_train.T)
    yyt_vec = yyt.flatten('F') # flatten YYT into a long concatenation of its cols
    
    test_samples, num_training_samp = embs_test.shape[0], yyt.shape[1]
    
    #solve: min_Wt || vec(Y^T Y) - ( Kron(P, Ut) ) vec(Wt)_2^2 || + lambda || vec(Wt) ||_2
    for lamb in list(np.arange(0, 10 , 0.5)):
        model = TrainRidge( X, yyt_vec, lamb)
        YYT_TestPred = Inference(
            model, X_test, similarity_shape = (test_samples, num_training_samp))
        mse = ReconstructKNN(
            yyt_pred = YYT_TestPred, #yyt_train = yyt_train,
            y_test = Y_test,  y_train = Y_train )
        print(lamb, mse)


if __name__ == '__main__':
    main()

