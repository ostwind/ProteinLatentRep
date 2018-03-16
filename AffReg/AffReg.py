from DataProcessing import * 
import scipy.linalg as sl
from scipy.spatial.distance import cosine    
from scipy.cluster.vq import whiten 
from sklearn.preprocessing import StandardScaler
    
def ReconstructY(YYT_test, y_train):
    ''' detailed in ar_reconstruction.m in supplementary code 
        reconstructs Y from predicted YY^T, reconstructs Y_test from linear comb of Y_train span 
    '''
    YYT_test = YYT_test.T
    #y_train =  y_train.T # (num_samples, num DNA) -> (num DNA, num samples)

    A = np.matmul( y_train.T, YYT_test)
    O = sl.orth(y_train) # 213 X 213 as many basis vectors as training examples 
    
    print( O.shape, A.shape, y_train.shape )
    c, _,_,_ = np.linalg.lstsq(O, y_train) # np.linalg.solve(O, y_train) #
    
    print(c.shape)
    #(213, 16382) (16382, 24) (213, 16382)

    ct, _,_,_ =  np.matmul( np.linalg.inv(c.T), A ) # np.linalg.lstsq(c.T, A) #
    
    print(y_train.shape, YYT_test.shape, A.shape, O.shape, c.shape,ct.shape, np.matmul(O, ct).shape )
    #(213, 16382) (213, 24) (16382, 24) (213, 213) (213, 16382) (213, 24) (213, 24)
    return np.matmul(O, ct)

def MSE(pred, label):
    return np.square(pred - label).mean()

def ReconstructKNN(yyt_pred,  yyt_train, y_test,  y_train):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5).fit( yyt_train )    
    distances, indices = neigh.kneighbors( yyt_pred )

    preds = [] 
    preds2 = []
    for index, (NN_indices, d) in enumerate( zip( indices, distances ) ):
        weights = d/np.sum(d)
        NN = y_train[NN_indices]
        
        y_pred2 = np.multiply( NN, weights[:, np.newaxis] )
        y_pred2 = np.sum(y_pred2, axis = 0)
        preds2.append(y_pred2)

    mse2 = MSE(np.array(preds2), y_test)
    
    print( mse2)
    return 0#np.mean(valid_losses)

def Inference(model, test_data, test_label):
    #print(test_data.shape) # (5112, 5120)
    Y_test_squared = model.predict(test_data) # 5112 
    Y_test_squared = np.array( Y_test_squared ).reshape((23, 209), order = 'F') # 24 X 213 
    # |test_samples| X |train_samples| where columns are  un-concatenated
    return Y_test_squared

def TrainRidge(X, y):
    #Y_lr W P_T
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=True, random_state=None,  tol=0.001)
    clf.fit(X, y) 
    return clf

def main():
    args = create_parser()
    pearson_file, emb_file, profiles = args.pearson_file, args.emb_file, args.profiles

    Y = np.genfromtxt(profiles, skip_header =1)[:,1:]
    Y = np.nan_to_num(Y).T # (407, 16382)
    Y = StandardScaler().fit_transform(Y)
    ynames = np.array(open(profiles, 'r').readline().strip().split()[1:])    
    
    Y_train, Y_test, trainprots, testprots = original_script_dataset_processing(
        ynames, Y, arg1="0", pearson_file=pearson_file)
    
    if 'kmers.csv' in emb_file:                                        
        embedding = pd.read_csv(emb_file)#, sep='\t')
    else:
        embedding = pd.read_csv(emb_file, sep='\t')
    
    #print(embedding.head())
    embedding.rename(columns={embedding.columns[0]:"name"},inplace=True)
    embedding.columns = ["{0}_le_col".format(x) for x in embedding.columns]
    embedding = get_labeled(embedding, "name_le_col")
    embedding = average_overlapping_embs(embedding, "name_le_col")
    
    Y_train, embs_train, trainprots_final = filter_embs(Y_train, trainprots, embedding)
    Y_test, embs_test, testprots_final = filter_embs(Y_test, testprots, embedding)
    Y_train, Y_test = StandardScaler().fit_transform(Y_train), StandardScaler().fit_transform(Y_test)
    embs_train, embs_test = StandardScaler().fit_transform(embs_train), StandardScaler().fit_transform(embs_test)

    #embs_train, embs_test = StandardScaler().fit_transform(embs_train), StandardScaler().fit_transform(embs_test)

    print(Y_train.shape, Y_test.shape, embs_train.shape, embs_test.shape)

    low_rank_dim = 60
    U, S, V = np.linalg.svd(Y_train, full_matrices=False)
    U, S, V = U[:, :low_rank_dim], np.diag(S[:low_rank_dim]), V[:low_rank_dim, :]
    Y_LowRank = np.matmul(np.matmul( U, S), V)
    
    #min_Wt || vec(Y^T Y) - ( Kron(P, Ut) ) vec(Wt)_2^2 || + lambda || vec(Wt) ||_2
    X = np.kron( embs_train, U)
    y =  np.matmul(Y_LowRank, Y_LowRank.T).flatten('F') # flatten YYT into a long concatenation of its cols
    model = TrainRidge( X, y)
    
    X_test = np.kron(embs_test, U) 
    YYT_TestPred = Inference(model, X_test, Y_test)
    closest_similarity_profiles = ReconstructKNN(
        yyt_pred = YYT_TestPred, yyt_train = np.matmul(Y_train,Y_train.T),
        y_test = Y_test,  y_train = Y_train )

if __name__ == '__main__':
    main()
