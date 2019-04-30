'''
Given a data set that has already been totally preprocessed, this code 
(1) splits it into k folds for cross-validation 
(2) tests different models with different parameters
'''

#TODO balance the data before training and see if it helps
#TODO the data given to us will be messy, so we should clean it beforehand as well
#TODO when performing feature selection, compare validation and insample error to check for over/underfitting
#TODO test with the data sent to us for leaderboard

import sys
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def get_cross_validation_folds(examples, k):
  kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
  return kf

def separate_features_and_target(examples, target_index, feature_indices):
  X = np.array([element[feature_indices] for element in examples])
  y = np.array([element[target_index] for element in examples])
  return X, y

def normalize_with_zscores(data):
  scaler = StandardScaler()
  scaler.fit(data)
  return scaler.transform(data)

def logistic_regression(examples, kfolds, target_index, feature_indices, params={}):
  lr = LogisticRegression(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    clf = lr.fit(X_train, y_train)
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
  return accs, precs, recs

def svm_(examples, kfolds, target_index, feature_indices, params={}):
  svc = LinearSVC(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    #X_train = normalize_with_zscores(X_train)
    clf = svc.fit(X_train, [int(i) for i in y_train])
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    #X_test = normalize_with_zscores(X_test)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
  return accs, precs, recs 

def random_forest(examples, kfolds, target_index, feature_indices, params={}):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    #X_train = normalize_with_zscores(X_train)
    clf = rf.fit(X_train, [int(i) for i in y_train])
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    #X_test = normalize_with_zscores(X_test)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
  return accs, precs, recs 

def process_input(filename):
  lines = [np.array([float(i) for i in l.strip().split(',')]) for l in open(filename, 'r').readlines()[1:]] #disconsidering header
  return np.array(lines)

if __name__ == '__main__':
  '''
  argv[1] => preprocessed data set
  argv[2] => hyperparameter K for k-fold cross-validation
  '''
  examples = process_input(sys.argv[1]) 
  kfolds = get_cross_validation_folds(examples, int(sys.argv[2]))

  ### test logistic regression model ### 
#   print 'LOGISTIC REGRESSION'
#   accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), params={'solver':'lbfgs', 'multi_class':'multinomial'})
#   print 'average accuracy for 5 folds', np.mean(accs)
#   print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
#   print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

#   ### test svm ###
#   print 'SVM'
#   accs, precs, recs = svm_(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), params={'random_state':0, 'tol':1e-5})
#   print 'average accuracy for 5 folds', np.mean(accs)
#   print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
#   print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  ### test random forest ###
  print 'RANDOM FOREST'
  accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  
