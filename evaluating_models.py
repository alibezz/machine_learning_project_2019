'''
Given a data set that has already been totally preprocessed, this code 
(1) splits it into k folds for cross-validation 
(2) tests different models with different parameters
'''

import sys
import numpy as np
from sklearn.model_selection import KFold 

RANDOM_STATE = 42

def get_cross_validation_folds(examples, k):
  kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
  return kf.split(examples)

if __name__ == '__main__':
  '''
  argv[1] => preprocessed data set
  argv[2] => hyperparameter K for k-fold cross-validation
  '''
  lines = open(sys.argv[1], 'r').readlines()[1:] #disconsidering header
  kfolds = get_cross_validation_folds(lines, int(sys.argv[2]))
  for train_index, test_index in kfolds:
    print 'TRAIN', train_index
    print 'TEST', np.array(lines)[test_index]
