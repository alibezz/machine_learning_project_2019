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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
SKIP = 2

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
    print confusion_matrix(y_test, y_pred).ravel()
  return accs, precs, recs, fmeasures

def logistic_regression_test_sampling(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  sampling_schemes = [None, 'over', 'under']
  best_scheme = ''; max_mean_fmeasure = -1
  for samp in sampling_schemes:
    print 'scheme =', samp
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

def core_testing(learning_method, examples, kfolds, target_index, feature_indices, value_range, param_name=None, params={}, sampling=None):
  best_value = None; max_mean_fmeasure = -1
  for value in value_range:
    tmp = params
    if param_name:
      tmp[param_name] = value
      print param_name, '=', value
    accs, precs, recs, fmeasures = learning_method(examples, kfolds.split(examples), target_index, feature_indices, params=params, sampling=sampling)
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      best_value = value
      max_mean_fmeasure = mean_overall_fmeasure
  return best_value
    
def logistic_regression_test_C(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  For small values of C, we increase the regularization strength, i.e., we create simple models that may underfit the data. 
  For big values of C, we lower the power of regularization, i.e., the model may overfit the data
  '''
  C_param_range = [0.001,0.01,0.1,1,10,100,1000,10000]
  return core_testing(logistic_regression, examples, kfolds, target_index, feature_indices, C_param_range, param_name='C', params=params, sampling=sampling)

def logistic_regression_test_class_weight(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different class weights handle imbalanced data differently
  '''
  class_weights = [None, 'balanced'] 
  return core_testing(logistic_regression, examples, kfolds, target_index, feature_indices, class_weights, param_name='class_weight',
                      params=params, sampling=sampling)

def logistic_regression_test_penalty(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different norms for optimization
  '''
  penalties = ['l1', 'l2']
  return core_testing(logistic_regression, examples, kfolds, target_index, feature_indices, penalties, param_name='penalty', params=params, sampling=sampling) 

def logistic_regression_test_solver(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  '''
  different solvers generate different solutions
  '''
  solvers = ['lbfgs', 'liblinear', 'sag'] #newton-cg is way too slow
  return core_testing(logistic_regression, examples, kfolds, target_index, feature_indices, solvers, param_name='solver', params=params, sampling=sampling)
  
def random_forest(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []; fmeasures = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    if sampling == 'over': #bootstrap training samples in the minority class 
      X_train, y_train = bootstrap_training(X_train, y_train)
    elif sampling == 'under': #reduce the number of samples in the majority class
      X_train, y_train = undersample_training(X_train, y_train)
    clf = rf.fit(X_train, [int(i) for i in y_train])
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
    fmeasures.append((get_fmeasure(results[0][0], results[1][0]), get_fmeasure(results[0][1], results[1][1])))
  return accs, precs, recs, fmeasures 

def random_forest_test_sampling(examples, kfolds, target_index, feature_indices):
  sampling_schemes = [None, 'over', 'under']
  best_scheme = ''; max_mean_fmeasure = -1
  for samp in sampling_schemes:
    print 'scheme =', samp
    accs, precs, recs, fmeasures = random_forest(examples, kfolds.split(examples), target_index, feature_indices, sampling=samp)
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      best_scheme = samp
      max_mean_fmeasure = mean_overall_fmeasure
  return best_scheme

def random_forest_test_n_estimators(examples, kfolds,  target_index, feature_indices, params={}, sampling=None):
  n_estimators = [5, 10, 25, 50, 75, 100]
  return core_testing(random_forest, examples, kfolds, target_index, feature_indices, n_estimators, param_name='n_estimators', params=params, sampling=sampling)

def random_forest_test_criterion(examples, kfolds,  target_index, feature_indices, params={}, sampling=None):
  criteria = ['gini', 'entropy']
  return core_testing(random_forest, examples, kfolds, target_index, feature_indices, criteria, param_name='criterion', params=params, sampling=sampling)

def random_forest_test_depth(examples, kfolds,  target_index, feature_indices, params={}, sampling=None):
  depths = [2, 4, 8, 16, 32, 64, None]
  return core_testing(random_forest, examples, kfolds, target_index, feature_indices, depths, param_name='max_depth', params=params, sampling=sampling)

def random_forest_test_max_features(examples, kfolds,  target_index, feature_indices, params={}, sampling=None):
  max_features = ['auto', 'sqrt', 'log2', None]
  return core_testing(random_forest, examples, kfolds, target_index, feature_indices, max_features, param_name='max_features', params=params, sampling=sampling)

def adaboost(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  ab =  AdaBoostClassifier(**params)
  precs = []; recs = []; accs = []; fmeasures = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    if sampling == 'over': #bootstrap training samples in the minority class 
      X_train, y_train = bootstrap_training(X_train, y_train)
    elif sampling == 'under': #reduce the number of samples in the majority class
      X_train, y_train = undersample_training(X_train, y_train)
    clf = ab.fit(X_train, [int(i) for i in y_train])
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
    fmeasures.append((get_fmeasure(results[0][0], results[1][0]), get_fmeasure(results[0][1], results[1][1])))
    print confusion_matrix(y_test, y_pred).ravel()
  return accs, precs, recs, fmeasures 

def adaboost_test_sampling(examples, kfolds, target_index, feature_indices):
  sampling_schemes = [None, 'over', 'under']
  best_scheme = ''; max_mean_fmeasure = -1
  for samp in sampling_schemes:
    print 'scheme =', samp
    accs, precs, recs, fmeasures = adaboost(examples, kfolds.split(examples), target_index, feature_indices, sampling=samp)
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      best_scheme = samp
      max_mean_fmeasure = mean_overall_fmeasure
  return best_scheme

def adaboost_test_tree_depth(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  tree_depths = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=8)]
  return core_testing(adaboost, examples, kfolds, target_index, feature_indices, tree_depths, param_name='base_estimator', params=params, sampling=sampling)

  best_depth = ''; max_mean_fmeasure = -1
  for depth in tree_depths:
    print 'depth =', depth
    accs, precs, recs, fmeasures = adaboost(examples, kfolds.split(examples), target_index, feature_indices, max_tree_depth=depth, sampling=sampling)
    print 'average accuracy for folds', np.mean(accs)
    print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
    print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
    mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
    print 'average fmeasure for both classes', mean_overall_fmeasure 
    if mean_overall_fmeasure > max_mean_fmeasure:
      best_depth = depth
      max_mean_fmeasure = mean_overall_fmeasure
  return best_depth

def adaboost_test_n_estimators(examples, kfolds, target_index, feature_indices, tree_depth=1, params={}, sampling=None):
  n_estimators = [10, 25, 50, 100, 200, 500]
  params['base_estimator'] = DecisionTreeClassifier(max_depth=tree_depth)
  return core_testing(adaboost, examples, kfolds, target_index, feature_indices, n_estimators, param_name='n_estimators', params=params, sampling=sampling)

def adaboost_test_boosting(examples, kfolds, target_index, feature_indices, tree_depth=1, params={}, sampling=None):
  boosting = ['SAMME', 'SAMME.R']
  params['base_estimator'] = DecisionTreeClassifier(max_depth=tree_depth)
  return core_testing(adaboost, examples, kfolds, target_index, feature_indices, boosting, param_name='algorithm', params=params, sampling=sampling)

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
  num_features = len(examples[0]) - 1 #one field is the target  
  ### test logistic regression model ###
  print 'LOGISTIC REGRESSION -- STANDARD'
  # accs, precs, recs, fmeasures = logistic_regression(examples, kfolds.split(examples), -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'average accuracy for folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])
  # mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
  # print 'average fmeasure for both classes', mean_overall_fmeasure 
  
  # print 'LOGISTIC REGRESSION -- VARYING SAMPLING SCHEME'
  # best_sampling = logistic_regression_test_sampling(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'Best sampling scheme =', best_sampling
  # print 'LOGISTIC REGRESSION -- VARYING C -- SAMPLING =', best_sampling
  # best_C = logistic_regression_test_C(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), sampling=best_sampling)
  # print 'Best C =', best_C
  # best_sampling = 'over'; best_C = 100
  # print 'LOGISTIC REGRESSION -- C =', best_C, '-- VARYING CLASS_WEIGHT -- SAMPLING =', best_sampling
  # best_class_weight = logistic_regression_test_class_weight(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]),
  #                                                           sampling=best_sampling, params={'C':best_C})
  # print 'Best class_weight =', best_class_weight
  # print 'LOGISTIC REGRESSION -- C =', best_C, '-- class_weight =', best_class_weight, '-- VARYING PENALTY -- SAMPLING =', best_sampling
  # best_penalty = logistic_regression_test_penalty(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), sampling=best_sampling,
  #                                                      params={'C':best_C, 'class_weight':best_class_weight})
  # print best_penalty
  # print 'LOGISTIC REGRESSION -- C =', best_C, '-- class_weight =', best_class_weight, '-- penalty =', best_penalty, '-- VARYING SOLVER -- SAMPLING =', best_sampling
  # #unfortunately, we have to use penalty = l2 regardless, as some solvers do not work with l1
  # best_solver = logistic_regression_test_solver(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), sampling=best_sampling, params={'C':best_C, 'class_weight':best_class_weight})
  # print 'Best solver =', best_solver

  # ### test random forest model ###
  # print 'RANDOM FOREST -- STANDARD'
  # accs, precs, recs, fmeasures = random_forest(examples, kfolds.split(examples), -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'average accuracy for folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  
  # mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
  # print 'average fmeasure for both classes', mean_overall_fmeasure
  # print 'RANDOM FOREST -- VARYING SAMPLING SCHEME'
  # best_sampling = random_forest_test_sampling(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'Best sampling scheme =', best_sampling
  
  # print 'RANDOM FOREST -- VARYING N_ESTIMATORS -- SAMPLING =', best_sampling
  # best_n = random_forest_test_n_estimators(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), sampling=best_sampling)
  # print 'Best number of estimators =', best_n
  # print 'RANDOM FOREST -- VARYING CRITERION -- N_ESTIMATORS =', best_n, '-- SAMPLING =', best_sampling
  # best_criterion = random_forest_test_criterion(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]),
  #                                               params={'n_estimators':best_n}, sampling=best_sampling)
  # print 'Best criterion =', best_criterion
  # print 'RANDOM FOREST -- VARYING DEPTH -- CRITERION =', best_criterion, '-- N_ESTIMATORS =', best_n, '-- SAMPLING =', best_sampling
  # best_depth = random_forest_test_depth(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]),
  #                                               params={'n_estimators':best_n, 'criterion':best_criterion}, sampling=best_sampling)
  # print 'Best depth =', best_depth
  # print 'RANDOM FOREST -- VARYING MAX FEATURES -- DEPTH =', best_depth, '-- CRITERION =', best_criterion, '-- N_ESTIMATORS =', best_n, '-- SAMPLING =', best_sampling
  # best_max_features = random_forest_test_max_features(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]),
  #                                               params={'n_estimators':best_n, 'criterion':best_criterion, 'max_depth':best_depth}, sampling=best_sampling)
  # print 'Best max features =', best_max_features

  ### test adaboost model ###
  # print 'ADABOOST WITH SHALLOW DECISION TREES -- STANDARD'
  # accs, precs, recs, fmeasures = adaboost(examples, kfolds.split(examples), -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'average accuracy for folds', np.mean(accs)
  # print 'average precision for class NO', np.mean([i[0] for i in precs]), 'average precision for class YES', np.mean([i[1] for i in precs])  
  # print 'average recall for class NO', np.mean([i[0] for i in recs]), 'average recall for class YES', np.mean([i[1] for i in recs])  
  # mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
  # print 'average fmeasure for both classes', mean_overall_fmeasure
  # # print 'ADABOOST WITH SHALLOW DECISION TREES -- VARYING SAMPLING SCHEME'
  # best_sampling = adaboost_test_sampling(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]))
  # print 'Best sampling scheme =', best_sampling
  # print 'ADABOOST WITH SHALLOW DECISION TREES -- VARYING TREE SHALLOWNESS -- SAMPLING =', best_sampling
  # best_tree_depth = adaboost_test_tree_depth(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), sampling=best_sampling)
  # print 'Best tree depth =', best_tree_depth
  best_tree_depth = 1
  best_sampling = 'over'
  best_n= 200
  best_boosting = 'SAMME'
  # print 'ADABOOST WITH SHALLOW DECISION TREES -- VARYING N_ESTIMATORS -- TREE_DEPTH =', best_tree_depth, '-- SAMPLING =', best_sampling
  # best_n = adaboost_test_n_estimators(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), tree_depth=best_tree_depth, sampling=best_sampling)
  #print 'ADABOOST WITH SHALLOW DECISION TREES -- VARYING BOOSTING ALGO -- N_ESTIMATORS =', best_n, '-- TREE_DEPTH =', best_tree_depth, '-- SAMPLING =', best_sampling
  #best_boosting = adaboost_test_boosting(examples, kfolds, -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]), tree_depth=best_tree_depth,
  #                                       params={'n_estimators':best_n}, sampling=best_sampling)
  accs, precs, recs, fmeasures = adaboost(examples, kfolds.split(examples), -1, np.array([i + SKIP for i in xrange(num_features - SKIP)]),
                                          sampling=best_sampling, params={'n_estimators':best_n, 'algorithm':best_boosting})
