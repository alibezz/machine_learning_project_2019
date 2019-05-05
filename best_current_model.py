'''
Given a data set that has already been totally preprocessed, this code 
(1) splits it into k folds for cross-validation 
(2) tests different models with different parameters
'''

import sys
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn.utils import resample

RANDOM_STATE = 42
SKIP = 0

def get_cross_validation_folds(k):
  kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
  return kf

def separate_features_and_target(examples, target_index, feature_indices):
  X = np.array([element[feature_indices] for element in examples])
  y = np.array([element[target_index] for element in examples])  
  return X, y

def bootstrap_training(training_examples, training_classes):
  lcc = Counter(training_classes).most_common()[-1] #lcc is the Least Common Class
  examples_in_lcc = [elem for i, elem in enumerate(training_examples) if training_classes[i] == lcc[0]]
  number_of_samples = len(training_examples) - 2*len(examples_in_lcc) #by doing so, you end up with the same number of samples per class
  new_samples = resample(examples_in_lcc, n_samples=number_of_samples, random_state=RANDOM_STATE)
  training_examples = np.concatenate((training_examples, new_samples))
  training_classes = np.concatenate((training_classes, [lcc[0] for i in new_samples]))
  return training_examples, training_classes

def get_fmeasure(precision, recall):
  return (2.*precision*recall)/(precision+recall)

def profile_errors_for_minority_class(examples, true_labels, predicted_labels):
  lcl = Counter(true_labels).most_common()[-1][0] #lcl is the Least Common Label, i.e., minority class
  incorrectly_predicted = []
  correctly_predicted = []
  for index, example in enumerate(examples):
    if true_labels[index] == lcl:
      if true_labels[index] != predicted_labels[index]:
        incorrectly_predicted.append(example)
      else:
        correctly_predicted.append(example)
  num_features = len(examples[0])
  averages_incorrect = []
  for i in xrange(num_features):
    vals = [elem[i] for elem in incorrectly_predicted]
    averages_incorrect.append(np.mean(vals))
  averages_correct = []
  for i in xrange(num_features):
    vals = [elem[i] for elem in correctly_predicted]
    averages_correct.append(np.mean(vals))
  feature_index = 0
  important_features = [1,5,7,8,9,34]
  for ai, ac in zip(averages_incorrect, averages_correct):
    if feature_index in important_features:
      print feature_index + SKIP, '- incorrect', ai, 'correct', ac
    feature_index += 1
  print '====================='
  
def random_forest(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []; fmeasures = []
  for train_index, test_index in kfolds:
    X_train, y_train = separate_features_and_target(examples[train_index], target_index, feature_indices)
    #bootstrap training samples in the minority class 
    X_train, y_train = bootstrap_training(X_train, y_train)
    clf = rf.fit(X_train, [int(i) for i in y_train])
    X_test, y_test = separate_features_and_target(examples[test_index], target_index, feature_indices)
    y_pred = clf.predict(X_test)
    results = precision_recall_fscore_support(y_test, y_pred)
    #profile_errors_for_minority_class(X_test, y_test, y_pred)
    precs.append(results[0])
    recs.append(results[1])
    accs.append(clf.score(X_test, y_test))
    fmeasures.append((get_fmeasure(results[0][0], results[1][0]), get_fmeasure(results[0][1], results[1][1])))
  return accs, precs, recs, fmeasures 

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

  print 'RANDOM FOREST'
  current_fmeasure = -1
  #greedy selection of features
  feature_indices = np.array([i + SKIP for i in xrange(num_features - SKIP)])
  current_indices = []
  #for index in feature_indices:
  #  current_indices.append(index)
  accs, precs, recs, fmeasures = random_forest(examples, kfolds.split(examples), -1, np.array([i for i in xrange(num_features)]),
                                               params={'n_estimators':1000, 'max_depth':32, 'criterion':'entropy', 'max_features':None}, sampling='over')
  mean_overall_fmeasure = (np.mean([i[0] for i in fmeasures]) + np.mean([i[1] for i in fmeasures]))/2
  print mean_overall_fmeasure
