import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, average_precision_score
from sklearn.model_selection import KFold 
from scipy.stats import spearmanr
import numpy as np  

def plot(x, y, labels, index, corr, auc):

    # plt.step(x, y, color='b', alpha=0.2,
    #      where='post')
    # plt.fill_between(x, y, step='post', alpha=0.2,
    #                 color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve')

    # plt.plot(x, y, color='darkorange', label='ROC curve ' )
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    
    one_indices = [i for i, x in enumerate(labels) if x ]
    zero_indices = [i for i, x in enumerate(labels) if not x ]
    
    plt.scatter(x[zero_indices], y[zero_indices], c='b')
    plt.scatter(x[one_indices], y[one_indices], c = 'r')

    plt.xlabel('True Probe Values')
    plt.ylabel('Predicted Values')
    #plt.ylim([0.0, 1.05])
    #plt.xlim([0.0, 1.0])
    plt.title(' profile%s, spearman correlation: %.2f' %(index, corr)) #op 1% {0:0.2f}'.format(average_precision))
    plt.savefig('./profile_plots/%s.png'%(index, ))
    plt.clf()

def EvalFold(y_pred, y_test):
    aucs, total_precision, spearman_corr = 0, 0, 0
    fold_stats = np.zeros(3)
    num_one_percent_most_intense_probes = int(len(y_test[0])/100)
    for index, (pred, truth) in enumerate( zip(y_pred, y_test)  ):
        # select top 1% of probe intensities in pred
        #pred_brightest_indices = np.argsort(pred)

        test_cutoff_val = truth[ np.argsort(truth)[-num_one_percent_most_intense_probes] ]
        
        # compute AUC  
        #test_brightest = truth[test_brightest_indices]
        #pred_brightest = pred[ pred_brightest_indices ]

        # intensities not found in top 1% of truth -> 0,  found in top 1% -> 1
        test_one_zero = [ 1 if x >= test_cutoff_val else 0 for x in truth ]
        test_one_zero = np.array(test_one_zero)

        fpr, tpr, thresholds = roc_curve(test_one_zero, pred, pos_label=1)
        fold_stats[0] +=  auc(fpr, tpr, )
        
        #precision, recall, _ = precision_recall_curve(test_one_zero, pred_brightest, pos_label = 1)
        # pred and truth have same ordering 
        fold_stats[1] += average_precision_score(test_one_zero, pred) 

        correlation, p_val = spearmanr( pred, truth )
        fold_stats[2] += correlation
        
        #plot( truth, pred, test_one_zero, index, correlation, auc(fpr, tpr, ) )
        
    return fold_stats#aucs, total_precision, spearman_corr 



