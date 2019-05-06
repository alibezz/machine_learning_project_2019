'''
Given CLEANED training and  test datasets, this code
runs our best model so far and gets its fmeasure 
'''

import sys
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn.utils import resample

RANDOM_STATE = 42
SKIP = 1

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

  
def random_forest(examples_training, examples_test, target_index, features_training, features_test, params={}, sampling=None):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []; fmeasures = []
  X_train, y_train = separate_features_and_target(examples_training, target_index, features_training)
  #bootstrap training samples in the minority class 
  X_train, y_train = bootstrap_training(X_train, y_train)
  clf = rf.fit(X_train, y_train)
  X_test = np.array([element[features_test] for element in examples_test])
  y_pred = clf.predict(X_test)
  with open('test_outputs.csv', 'w') as f:
    f.write('id_num,quidditch_league_player\n')
    for index, elem in enumerate(y_pred):
      if int(elem) == 0:
        f.write(str(index+1) + str(',') + 'NO\n')
      else:
        f.write(str(index+1) + str(',') + 'YES\n')

def process_input(filename):
  lines = open(filename, 'r').readlines()
  print lines[0]
  print lines[1]
  examples = [np.array([float(i) for i in l.strip().split(',')]) for l in lines[1:]]
  header = lines[0]
  features = header.strip().split(',')
  return np.array(examples), np.array(features)

def determine_feature_intersection(features_training, features_test):
  training_dict = {}
  test_dict = {}
  for index, elem in enumerate(features_training):
    training_dict[elem] = index
  for index, elem in enumerate(features_test):
    test_dict[elem] = index
  key_intersection = list(set(training_dict.keys()) & set(test_dict.keys()))
  training_indices = []
  test_indices = []
  for relevant_key in key_intersection:
    training_indices.append(training_dict[relevant_key])
    test_indices.append(test_dict[relevant_key])
  return sorted(training_indices), sorted(test_indices)

def get_cross_validation_folds(k):
  kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
  return kf

def random_forest2(examples, kfolds, target_index, feature_indices, params={}, sampling=None):
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


if __name__ == '__main__':
  '''
  argv[1] => training dataset
  argv[2] => test dataset
  '''
  examples_training, features_training = process_input(sys.argv[1])
  examples_test, features_test = process_input(sys.argv[2])
  relevant_training_feature_ids, relevant_test_feature_ids = determine_feature_intersection(features_training, features_test)
  print 'RANDOM FOREST'
  random_forest(examples_training, examples_test, -1, relevant_training_feature_ids, relevant_test_feature_ids, params={'n_estimators':5, 'max_depth':32, 'criterion':'entropy', 'max_features':None}, sampling='over')
