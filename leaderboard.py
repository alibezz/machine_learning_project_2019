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
SKIP = 2

def separate_features_and_target(examples, target_index, feature_indices):
  X = np.array([element[feature_indices][0] for element in examples])
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

  
def random_forest(examples_training, examples_test, target_index, feature_indices, params={}, sampling=None):
  rf = RandomForestClassifier(**params)
  precs = []; recs = []; accs = []; fmeasures = []
  X_train, y_train = separate_features_and_target(examples_training, target_index, feature_indices)
  print X_train[0]
  #bootstrap training samples in the minority class 
  X_train, y_train = bootstrap_training(X_train, y_train)
  clf = rf.fit(X_train, y_train)
  X_test = np.array([i[1:] for i in examples_test])
  y_pred = clf.predict(X_test)
  with open('leaderboard_output.txt', 'w') as f:
    f.write('id_num,quidditch_league_player\n')
    for index, elem in enumerate(y_pred):
      f.write(str(index+1) + str(',') + str(int(elem)) + '\n')

def process_input(filename):
  lines = [np.array([float(i) for i in l.strip().split(',')]) for l in open(filename, 'r').readlines()[1:]] #disconsidering header
  return np.array(lines)

if __name__ == '__main__':
  '''
  argv[1] => training dataset
  argv[2] => test dataset
  '''
  examples_training = process_input(sys.argv[1])
  examples_test = process_input(sys.argv[2])
  leaderboard_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59, 62, 64, 65, 66, 69, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 91, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128]
  num_features = len(leaderboard_features) - 1 #one field is the target

  print 'RANDOM FOREST'
  random_forest(examples_training, examples_test, -1, np.array([leaderboard_features]), params={'n_estimators':10000, 'max_depth':128, 'criterion':'gini', 'max_features':None}, sampling='over')
