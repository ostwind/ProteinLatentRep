import os

import numpy as np
from util import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

'''
we do protein family classification as the downstream task to Seq2Vec
the paper did cross validation and gridsearch (for SVM classifier),
'''
write_pred_target('ws3_size100_win25_e5.d2v')
X_train, X_test, y_train, y_test = train_test_split()
#k_fold = kFold(n_splits = 10)
#cross_val_scores = [svc.fit(matrix[train], target[train]).score(matrix[test], target[test]) for train, test in k_fold.split(matrix)]
#cross_val_score(svc, matrix, target, cv=k_fold, n_jobs=-1)

# Cs = np.logspace(-1, 3, 5)
# clf = GridSearchCV(estimator = svm.SVC(), param_grid = dict(C=Cs), n_jobs = -1)
# clf.fit(matrix[:5000], target[:5000])
# print(clf.best_score_, clf.best_estimator_.C)

classif = svm.SVC(C = 2)
classif.fit(X_train, y_train)
pred = classif.predict(X_test)
print(accuracy_score(y_test, pred))

#print(np.logspace(-1, 5, 3))
