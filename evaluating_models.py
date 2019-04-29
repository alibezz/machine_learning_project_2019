'''
Given a data set that has already been totally preprocessed, this code 
(1) splits it into k folds for cross-validation 
(2) tests different models with different parameters
'''

#TODO balance the data before training and see if it helps

import sys
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

RANDOM_STATE = 42

def get_cross_validation_folds(examples, k):
  kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
  return kf.split(examples)

def separate_features_and_target(examples, target_index, feature_indices):
  X = np.array([element[feature_indices] for element in examples])
  y = np.array([element[target_index] for element in examples])
  return X, y

def logistic_regression(examples, kfolds, target_index, feature_indices, params={}):
  lr = LogisticRegression(**params)
  precs = []; recs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    clf = lr.fit(X_train, y_train)
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    print 'true', y_test
    print 'pred', y_pred
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    print precs
    print recs
    print results
    break
    #print 'TRAIN', examples[train_index]
    #print 'TEST', test_index

def process_input(filename):
  lines = [np.array([int(i) for i in l.strip().split(',')]) for l in open(filename, 'r').readlines()[1:]] #disconsidering header
  return np.array(lines)

if __name__ == '__main__':
  '''
  argv[1] => preprocessed data set
  argv[2] => hyperparameter K for k-fold cross-validation
  '''
  examples = process_input(sys.argv[1]) 
  kfolds = get_cross_validation_folds(examples, int(sys.argv[2]))

  ### test logistic regression model ### 
  fmeasures = logistic_regression(examples, kfolds, -1, np.array([2, 3, 4]), {'solver':'lbfgs', 'multi_class':'multinomial'})
