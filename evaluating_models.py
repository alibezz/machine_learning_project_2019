'''
Given a data set that has already been totally preprocessed, this code 
(1) splits it into k folds for cross-validation 
(2) tests different models with different parameters
'''

#TODO balance the data before training and see if it helps
#TODO the data given to us will be messy, so we should clean it beforehand as well
#TODO Implement PCA to reduce features, but also to create features
#TODO Implement different normalization techniques
#TODO when performing feature selection, compare validation and insample error to check for over/underfitting
#TODO test with the data sent to us for leaderboard
#TODO Transform just a few features (Linda's suggestion)
#TODO See how the features correlate with the target (someone wrote it on piazza)
#TODO Discuss bias and variance (insample vs. validation error plots) when doing feature creation/selection (Linda liked it)
#TODO Use Adaboost or some ensemble technique

import sys
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.utils import resample

RANDOM_STATE = 42

def get_cross_validation_folds(k):
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

def bootstrap_training(training_examples, training_classes):
  lcc = Counter(training_classes).most_common()[-1] #lcc is the Least Common Class
  examples_in_lcc = [elem for i, elem in enumerate(training_examples) if training_classes[i] == lcc[0]]
  number_of_samples = len(training_examples) - 2*len(examples_in_lcc) #by doing so, you end up with the same number of samples per class
  new_samples = resample(examples_in_lcc, n_samples=number_of_samples, random_state=RANDOM_STATE)
  training_examples = np.concatenate((training_examples, new_samples))
  training_classes = np.concatenate((training_classes, [lcc[0] for i in new_samples]))
  return training_examples, training_classes

def undersample_training(training_examples, training_classes):
  mcc = Counter(training_classes).most_common()[0] #lcc is the Most Common Class
  lcc = Counter(training_classes).most_common()[-1]
  examples_in_mcc = [elem for i, elem in enumerate(training_examples) if training_classes[i] == mcc[0]]
  number_of_samples = len(training_examples) - len(examples_in_mcc) #we undersample to the same number of samples in the least common class
  new_samples = resample(examples_in_mcc, n_samples=number_of_samples, random_state=RANDOM_STATE)
  training_examples = np.concatenate((new_samples, [elem for i, elem in enumerate(training_examples) if training_classes[i] == lcc[0]]))
  training_classes = np.concatenate(([mcc[0] for i in new_samples], [lcc[0] for i in xrange(number_of_samples)]))
  return training_examples, training_classes

def logistic_regression(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  lr = LogisticRegression(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    if sampling == 'over': #bootstrap training samples in the minority class 
      X_train, y_train = bootstrap_training(X_train, y_train)
    elif sampling == 'under': #reduce the number of samples in the majority class
      X_train, y_train = undersample_training(X_train, y_train)
    clf = lr.fit(X_train, y_train)
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
  return accs, precs, recs

def svm_(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  svc = LinearSVC(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    if sampling == 'over': #bootstrap training samples in the minority class 
      X_train, y_train = bootstrap_training(X_train, y_train)
    elif sampling == 'under': #reduce the number of samples in the majority class
      X_train, y_train = undersample_training(X_train, y_train)
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

def random_forest(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    if sampling == 'over': #bootstrap training samples in the minority class 
      X_train, y_train = bootstrap_training(X_train, y_train)
    elif sampling == 'under': #reduce the number of samples in the majority class
      X_train, y_train = undersample_training(X_train, y_train)
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
  kfolds = get_cross_validation_folds(int(sys.argv[2]))
  
  ### test logistic regression model ### 
  print 'LOGISTIC REGRESSION -- STANDARD'
  accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]))
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS'
  accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'})
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS -- UNDERSAMPLING'
  accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'}, sampling='under')
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS -- OVERSAMPLING'
  accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'}, sampling='over')
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

#   ### test svm ###
#   print 'SVM'
#   accs, precs, recs = svm_(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), sampling='over')# params={'random_state':0, 'tol':1e-5})
#   print 'average accuracy for 5 folds', np.mean(accs)
#   print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
#   print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  ### test random forest ###
  print 'RANDOM FOREST -- STANDARD'
  accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]))
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH'
  accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH -- UNDERSAMPLING'
  accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), sampling='under', params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH -- OVERSAMPLING'
  accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]), sampling='over', params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  print 'average accuracy for 5 folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  
