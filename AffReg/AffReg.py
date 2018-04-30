from DataProcessing import * 
from scipy.stats import pearsonr 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge, Lasso 
import pickle        
from evaluation import * 

def norm(array_like):
    #if len( array_like.shape ) == 2:
        # norm row-wise
    return normalize(array_like, axis = 1) #np.divide( array_like, np.linalg.norm( array_like, axis = 1 )[:, np.newaxis])
    #return array_like/np.linalg.norm(array_like)

def ReconstructKNN(yyt_pred,  y_test,  y_train):
    preds = []
    #yyt_pred_normed = norm(yyt_pred) #, training_normed, test_normed = norm(yyt_pred), norm(y_train), norm(y_test)

    for index, test_sample_sim in enumerate(yyt_pred):
        y_pred = np.matmul( y_train.T, test_sample_sim )
        preds.append(y_pred)

    fold_stats = EvalFold(  normalize( np.array(preds), axis = 1) , normalize( y_test, axis = 1) )
    return fold_stats 

def TrainRidge_Inference(lamb, X_train, X_test, yyt, U):
    clf = Ridge(alpha = lamb, copy_X=False,
    fit_intercept=True, normalize = True, random_state=None,  tol=0.001,
    #precompute = True 
    )
    similarity_shape = (X_test.shape[0], X_train.shape[0])
    
    X_train = np.kron(X_train, U )
    clf.fit(  X_train , yyt ) 
    #print( '%.2f' %np.linalg.norm(clf.coef_ ) )
    X_test = np.kron(X_test, U )
    
    Y_test_squared = clf.predict( X_test) 
    Y_test_squared =  Y_test_squared.reshape(similarity_shape) # 24 X 213 

    return Y_test_squared


def kfold_validation(X, y, num_folds = 10 ):
    kf =KFold(n_splits = num_folds)
    
    print( X.shape[0] - X.shape[0]//num_folds,
     ' training samples || ', X.shape[0], ' total samples'   )
    for lamb in np.linspace(0, 30, num=40):
        fold_stats =  np.zeros(3)

        for index, (train_index, test_index) in enumerate(kf.split(X)) :
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            #Y_train, Y_test = norm(Y_train), norm(Y_test) 
            U = SVD(Y_train)
            
            yyt =  np.matmul(Y_train, Y_train.T)
            yyt_vec = yyt.flatten() # flatten YYT into a long concatenation of its cols    
            yyt_pred = TrainRidge_Inference( lamb, X_train, X_test, yyt_vec, U )
            
            fold_stats += ReconstructKNN(
                yyt_pred = yyt_pred, 
                y_test = Y_test,  y_train = Y_train )
        
        fold_stats /= num_folds*Y_test.shape[0]
        print('%.2f,    auc: %.3f   avg_prec: %.3f    spearman: %.3f' 
        %(lamb, fold_stats[0], fold_stats[1], fold_stats[2] )  )

    #return total_stats / X.shape[0]

def main():
    X, Y, prots = LoadData()
    Y = norm(Y)
    # X = normalize(X, axis = 1)
    
    pickle.dump( Y, open( "64kmer.p", "wb" ) )
    # kfold_validation( X, Y)
    
def main2():
   df = pd.read_csv('output_list.txt', sep=" ", header=None)

if __name__ == '__main__':
    main()


def ParameterSearch(X_train, X_test, Y_train, Y_test,
     yyt_vec, similarity_shape ):
    #solve: min_Wt || vec(Y^T Y) - ( Kron(P, Ut) ) vec(Wt)_2^2 || + lambda || vec(Wt) ||_2
    
    for lamb in np.linspace(0, 20, num=20):
        YYT_TestPred = TrainRidge_Inference( X_train, yyt_vec, lamb, X_test, similarity_shape)
        fold_stats = ReconstructKNN(
            yyt_pred = YYT_TestPred, 
            y_test = Y_test,  y_train = Y_train )
        
        fold_stats /= Y_test.shape[0]
        #print( lamb, auc_sum/num_samples, pr_sum/num_samples, spearman_sum/num_samples )
        print('%.3f,    auc: %.3f   avg_prec: %.3f    spearman: %.3f' 
        %(lamb, fold_stats[0], fold_stats[1], fold_stats[2] )  )


