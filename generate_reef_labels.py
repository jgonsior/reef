#!/usr/bin/env python
# coding: utf-8
from pprint import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

parser = argparse.ArgumentParser()

# folder data/$data_set/{train,test,validation}.csv?
# create small script data/$data_set/splitData.py !!
# create dataset deco_small, deco_full -> recheck why I have only 40 featuren instead of 159! was it the result of feature selection? kBest 40?
parser.add_argument('--data_set', required=True)
parser.add_argument(
    '--strategy',
    required=True,
    help="Possible Values: uncertainty|random|committee|boundary")
parser.add_argument('--nLearningIterations', type=int, default=15)
parser.add_argument('--nQueriesPerIteration', type=int, default=150)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--mergedLabels', action='store_true')
parser.add_argument('--output', default="results/default")
parser.add_argument('--cores', type=int, default=1)
parser.add_argument('--start_size', type=float)
parser.add_argument('--random_seed', type=int, default=23)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

np.random.seed(config.random_seed)
random.seed(config.random_seed)

dataset = 'imdb'

from data.loader import DataLoader
dl = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground, _, _, _ = dl.load_data(
    data_path='./data/imdb/budgetandactors.txt')

pprint(train_primitive_matrix)

exit(-3)
# # Reef Steps
# Reef generates heuristics in an iterative manner, with each iteration consisting of the following steps:
# 1. Synthesize Heuristics
# 2. Prune Heuristics
# 3. Verify Heuristics
#
# In this tutorial, we go through the three stages of Reef individually and then repeat the process iteratively.

# In the cell below, we run a single iteration by calling the `run_synthesizer` function. We pass in the primitive matrices for the `train` and `val` sets, along with ground truth labels for `val`. While we also pass in ground truth labels for `train`, this is solely for evaluation purposes.
#
# `max_cardinality` is the maximum number of primitives a heuristic takes as input, `keep` is how many heuristics the pruner should select (3 for the first iteration, 1 after that) and `model` is the type of heuristic to generate, in this case, `decision_tree`.
#
# _This cell does not output anything, only saves values in HeuristicGenerator._

# In[3]:

from program_synthesis.heuristic_generator import HeuristicGenerator

hg = HeuristicGenerator(train_primitive_matrix,
                        val_primitive_matrix,
                        val_ground,
                        train_ground,
                        b=0.5)
hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model='dt')

# ## 1. Synthesize Heuristics
# We start by generating all possible heuristics based on the labeled, validation set that take in a single feature (i.e. word for this example) as input.
#
# For this example, we use decision trees with maximum depth 1 (`dt`) as our heuristic form. This translates to checking whether a certain word exists or does not exist in the text to assign a label. We first generate all possible heuristics that take a single feature in as input.

# In[4]:

from program_synthesis.synthesizer import Synthesizer
syn = Synthesizer(val_primitive_matrix, val_ground, b=0.5)

heuristics, feature_inputs = syn.generate_heuristics('nn', 1)
print("Total Heuristics Generated: ", np.shape(heuristics)[1])

# For each generated heuristic, we find an associated $\beta$ value.  This corresponds to defining a region of **low confidence** labels, which the heuristic will abstain for, while labeling the rest of the datapoints as $1$ or $-1$.

# In[5]:

optimal_betas = syn.find_optimal_beta(heuristics[0], val_primitive_matrix,
                                      feature_inputs[0], val_ground)
plt.hist(optimal_betas, range=(0, 0.5))
plt.xlabel('Beta Values')

# ## 2. Prune Heuristics
# In the first iteration, we simply pick the 3 heuristics that perform the best on the labeled validation set.

# In[6]:

top_idx = hg.prune_heuristics(heuristics, feature_inputs, keep=3)
print('Features chosen heuristics are based on: ', top_idx)

# In subsequent iterations (step 4), we weight the Jaccard score (overlap of how many datapoints in the train set receive labels and how many are labeled by existing heuristics) and F1 score equally. We demonstrate this with a toy vector of previously labeled data.

# ## 3. Verify Heuristics
# In this step, we use the labels the heuristics assign to the **unlabeled train set** to estimate heuristic accuracies and assign probabilistic training labels to the same set accordingly (see [snorkel.stanford.edu](http://snorkel.stanford.edu) for more details).

# In[7]:

from program_synthesis.verifier import Verifier
verifier = Verifier(hg.L_train, hg.L_val, val_ground, has_snorkel=False)

verifier.train_gen_model()
verifier.assign_marginals()

# We visualize what these labels look like. Note that with a single iteration, none of the datapoints receive a probabilistic label greater than 0.5, but this is fixed after running the process iteratively (Step 4). __These labels are then used to train an end model, such as an LSTM, and not used as final predictions.__

# In[8]:

plt.hist(verifier.train_marginals)
plt.title('Training Set Probabilistic Labels')

# Since we do not have access to ground truth labels for the train set, we use the distribution of labels for the labeled validation set to decide what feedback to pass to the synthesizer. We pass datapoints with low confidence (labels near 0.5, i.e. equal probability of being +1 or -1) to the synthesizer

# In[9]:

plt.hist(verifier.val_marginals)
plt.title('Validation Set Probabilistic Labels')
feedback_idx = verifier.find_vague_points(gamma=0.1, b=0.5)
print('Percentage of Low Confidence Points: ',
      np.shape(feedback_idx)[0] / float(np.shape(val_ground)[0]))

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

hg = HeuristicGenerator(train_primitive_matrix,
                        val_primitive_matrix,
                        val_ground,
                        train_ground,
                        b=0.5)
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

# In the plots above, we show the distribution of probabilistic labels Reef assigns to the training set in the first few iterations.
#
# Next, we look at the accuracy and coverage of labels assigned to the training set in the _last_ iteration. The coverage is the percentage of training set datapoints that receive at least one label from the generated heuristics.

# In[13]:

plt.hist(training_marginals[-1], bins=10, range=(0.0, 1.0))
plt.title('Final Distribution')

print("Program Synthesis Train Accuracy: ", training_accuracy[-1])
print("Program Synthesis Train Coverage: ", training_coverage[-1])
print("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])

# ### Save Training Set Labels
# We save the training set labels Reef generates that we use in the next notebook to train a simple LSTM model.

# In[14]:

filepath = './data/' + dataset
np.save(filepath + '_reef.npy', training_marginals[-1])
