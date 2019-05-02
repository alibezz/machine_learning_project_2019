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

def get_fmeasure(precision, recall):
  return (2.*precision*recall)/(precision+recall)

def logistic_regression(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  lr = LogisticRegression(**params)
  precs = []; recs = []; accs = []; fmeasures = []
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
    fmeasures.append((get_fmeasure(results[0][0], results[1][0]), get_fmeasure(results[0][1], results[1][1])))
  return accs, precs, recs, fmeasures

def logistic_regression_test_sampling(examples, kfolds, target_index, feature_indices):
  sampling_schemes = [None, 'over', 'under']
  best_scheme = ''; max_mean_fmeasure = -1
  for samp in sampling_schemes:
    accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), target_index, feature_indices, sampling=samp)
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      best_scheme = samp
      max_mean_fmeasure = mean_overall_fmeasure
  return best_scheme
    
def logistic_regression_test_C(examples, kfolds, target_index, feature_indices, sampling=None):
  '''
  For small values of C, we increase the regularization strength, i.e., we create simple models that may underfit the data. 
  For big values of C, we lower the power of regularization, i.e., the model may overfit the data
  '''
  C_param_range = [0.001,0.01,0.1,1,10,100,1000,10000] #best was C = 100
  best_C = -1; max_mean_fmeasure = -1
  for C_ in C_param_range:
    accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), target_index, feature_indices, params={'C':C_}, sampling=sampling)
    print 'C =', C_
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      max_mean_fmeasure = mean_overall_fmeasure
      best_C = C_
  return best_C

def logistic_regression_test_class_weight(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different class weights handle imbalanced data differently
  '''
  class_weights = [None, 'balanced'] 
  best_class_weight = ''; max_mean_fmeasure = -1
  for cw in class_weights:
    tmp = params
    tmp['class_weight'] = cw
    accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), target_index, feature_indices, params=tmp, sampling=sampling)
    print 'class weight =', cw
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      max_mean_fmeasure = mean_overall_fmeasure
      best_class_weight = cw
  return best_class_weight

def logistic_regression_test_penalty(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different norms for optimization
  '''
  penalties = ['l1', 'l2'] 
  best_penalty = ''; max_mean_fmeasure = -1
  for penalty_ in penalties:
    tmp = params
    tmp['penalty'] = penalty_
    accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), target_index, feature_indices, params=tmp, sampling=sampling)
    print 'penalty =', penalty_
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      max_mean_fmeasure = mean_overall_fmeasure
      best_penalty = penalty_
  return best_penalty

def logistic_regression_test_solver(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different solvers generate different solutions
  '''
  solvers = ['lbfgs', 'liblinear', 'sag'] #newton-cg is way too slow
  best_solver = ''; max_mean_fmeasure = -1
  for solver_ in solvers:
    tmp = params
    tmp['solver'] = solver_
    accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), target_index, feature_indices, params=tmp, sampling=sampling)
    print 'solver =', solver_
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      max_mean_fmeasure = mean_overall_fmeasure
      best_solver = solver_
  return best_solver
  
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
  accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]))
  print 'average accuracy for folds', np.mean(accs)
  print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
  mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
  print 'average fmeasure for both classes', mean_overall_fmeasure 
  
  print 'LOGISTIC REGRESSION -- VARYING SAMPLING SCHEME'
  best_sampling = logistic_regression_test_sampling(examples, kfolds, -1, np.array([i for i in xrange(113)]))
  print 'Best sampling scheme =', best_sampling
  print 'LOGISTIC REGRESSION -- VARYING C -- SAMPLING =', best_sampling
  best_C = logistic_regression_test_C(examples, kfolds, -1, np.array([i for i in xrange(113)]), sampling=best_sampling)
  print 'Best C =', best_C
  print 'LOGISTIC REGRESSION -- C =', best_C, '-- VARYING CLASS_WEIGHT -- SAMPLING =', best_sampling
  best_class_weight = logistic_regression_test_class_weight(examples, kfolds, -1, np.array([i for i in xrange(113)]), sampling=best_sampling, params={'C':best_C})
  print 'Best class_weight =', best_class_weight
  print 'LOGISTIC REGRESSION -- C =', best_C, '-- class_weight =', best_class_weight, '-- VARYING PENALTY -- SAMPLING =', best_sampling
  best_penalty = logistic_regression_test_penalty(examples, kfolds, -1, np.array([i for i in xrange(113)]), sampling=best_sampling,
                                                       params={'C':best_C, 'class_weight':best_class_weight})
  print best_penalty
  print 'LOGISTIC REGRESSION -- C =', best_C, '-- class_weight =', best_class_weight, '-- penalty =', best_penalty, '-- VARYING SOLVER -- SAMPLING =', best_sampling
  #unfortunately, we have to use penalty = l2 regardless, as some solvers do not work with l1
  best_solver = logistic_regression_test_solver(examples, kfolds, -1, np.array([i for i in xrange(113)]), sampling=best_sampling, params={'C':best_C, 'class_weight':best_class_weight})
  print 'Best solver =', solver
  
  # accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]))
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS'
  # accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'})
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS -- UNDERSAMPLING'
  # accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'}, sampling='under')
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'LOGISTIC REGRESSION -- SOLVER/MULTICLASS TWEAKS -- OVERSAMPLING'
  # accs, precs, recs = logistic_regression(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'solver':'lbfgs', 'multi_class':'multinomial'}, sampling='over')
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  ### test random forest ###
  # print 'RANDOM FOREST -- STANDARD'
  # accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]))
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH'
  # accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH -- UNDERSAMPLING'
  # accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), sampling='under', params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  

  # print 'RANDOM FOREST -- N_ESTIMATORS/MAX_DEPTH -- OVERSAMPLING'
  # accs, precs, recs = random_forest(examples, kfolds.split(examples), -1, np.array([i for i in xrange(113)]), sampling='over', params={'n_estimators':100, 'max_depth':2, 'random_state':0})
  # print 'average accuracy for 5 folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  
