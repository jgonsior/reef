#!/usr/bin/env python
# coding: utf-8
import argparse
import random
import sys
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

from data.loader import DataLoader
from deco_helper.functions import splitFeatures
from program_synthesis.heuristic_generator import HeuristicGenerator
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier

parser = argparse.ArgumentParser()

# folder data/$data_set/{train,test,validation}.csv?
# create small script data/$data_set/splitData.py !!
# create dataset deco_small, deco_full -> recheck why I have only 40 featuren instead of 159! was it the result of feature selection? kBest 40?
parser.add_argument('--dataset_path', required=True)
parser.add_argument('--nLearningIterations', type=int, default=15)
parser.add_argument('--nQueriesPerIteration', type=int, default=150)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--mergedLabels', action='store_true')
parser.add_argument('--output', default="results/default")
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--start_size', type=float)
parser.add_argument('--random_seed', type=int, default=23)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

np.random.seed(config.random_seed)
random.seed(config.random_seed)

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

if config.dataset_path != 'imdb':
    train = pd.read_csv(config.dataset_path + '/6_train.csv')
    test = pd.read_csv(config.dataset_path + '/6_test.csv')
    val = pd.read_csv(config.dataset_path + '/6_val.csv')

    X_train, meta_train, Y_train, label_encoder_train = splitFeatures(train)
    X_test, meta_test, Y_test, label_encoder_test = splitFeatures(test)
    X_val, meta_val, Y_val, label_encoder_val = splitFeatures(val)

    n_classes = max(len(label_encoder_val.classes_),
                    len(label_encoder_test.classes_),
                    len(label_encoder_train.classes_))
else:
    dl = DataLoader()
    X_train, X_val, X_test, Y_train, Y_val, Y_test, _, _, _ = dl.load_data(
        data_path='./data/imdb/budgetandactors.txt')
    n_classes = 2

#  pprint(X_train)
#  pprint(Y_train)

#  pprint(X_test)
#  pprint(Y_test)

#  pprint(X_val)
#  pprint(Y_val)

# ## 4. Repeat Iterative Process of Generating Heuristics
# We repeat this process of synthesizing, pruning, and verifying heuristics iteratively. In this example, we generate 25 total heuristics.
#
# _Note that this process wil take a few minutes to run since heuristic generation is not currently optimized in any manner_

# In[10]:

validation_accuracy = []
training_accuracy = []
validation_coverage = []
training_coverage = []

training_marginals = []
idx = None

hg = HeuristicGenerator(X_train, X_val, Y_val, Y_train, b=1 / n_classes)
plt.figure(figsize=(12, 6))
for i in range(3, 26):
    if (i - 2) % 5 == 0:
        print("Running iteration: ", str(i - 2))

    #Repeat synthesize-prune-verify at each iterations
    if i == 3:
        hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
    else:
        hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
    hg.run_verifier()

    #Save evaluation metrics
    va, ta, vc, tc = hg.evaluate()
    validation_accuracy.append(va)
    training_accuracy.append(ta)
    training_marginals.append(hg.vf.train_marginals)
    validation_coverage.append(vc)
    training_coverage.append(tc)

    #Plot Training Set Label Distribution
    if i <= 8:
        plt.subplot(2, 3, i - 2)
        plt.hist(training_marginals[-1], bins=10, range=(0.0, 1.0))
        plt.title('Iteration ' + str(i - 2))
        plt.xlim([0.0, 1.0])
        plt.ylim([0, 825])

    #Find low confidence datapoints in the labeled set
    hg.find_feedback()
    idx = hg.feedback_idx

    #Stop the iterative process when no low confidence labels
    if idx == []:
        break
plt.tight_layout()
plt.show()

# In the plots above, we show the distribution of probabilistic labels Reef assigns to the training set in the first few iterations.
#
# Next, we look at the accuracy and coverage of labels assigned to the training set in the _last_ iteration. The coverage is the percentage of training set datapoints that receive at least one label from the generated heuristics.

# In[13]:
plt.hist(training_marginals[-1], bins=10, range=(0.0, 1.0))
plt.title('Final Distribution')
plt.show()

print("Program Synthesis Train Accuracy: ", training_accuracy)
print("Program Synthesis Train Coverage: ", training_coverage)
print("Program Synthesis Validation Accuracy: ", validation_accuracy)

# ### Save Training Set Labels
# We save the training set labels Reef generates that we use in the next notebook to train a simple LSTM model.

# In[14]:

filepath = './data/'
np.save(filepath + '_reef.npy', training_marginals[-1])
