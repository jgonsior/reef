#!/usr/bin/env python
# coding: utf-8

# In[2]:

#  get_ipython().run_line_magic('load_ext', 'autoreload')
#  get_ipython().run_line_magic('autoreload', '2')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import *
from lstm.imdb_lstm import *

import matplotlib.pyplot as plt
#  get_ipython().run_line_magic('matplotlib', 'inline')

# # Load Dataset
# We reload the dataset with the plain text plots and the labels that reef generated

# In[3]:

dataset = 'imdb'

from data.loader import DataLoader
dl = DataLoader()
_, _, _, train_ground, val_ground, test_ground, train_text, val_text, test_text = dl.load_data(
    dataset=dataset)
train_reef = np.load('./data/imdb_reef.npy')

# # Train an LSTM Model
# We now train a simple LSTM model with the labels generated by Reef. The following hyperparameter search is simplistic, and a more fine-tuned search and a more complex model can improve performance!
#
# __Note that this takes ~1 hour to run on CPU__

# In[3]:

f1_all = []
pr_all = []
re_all = []
val_acc_all = []

bs_arr = [64, 128, 256]
n_epochs_arr = [5, 10, 25]

for bs in bs_arr:
    for n in n_epochs_arr:
        y_pred = lstm_simple(train_text,
                             train_reef,
                             val_text,
                             val_ground,
                             bs=bs,
                             n=n)
        predictions = np.round(y_pred)

        val_acc_all.append(
            np.sum(predictions == val_ground) / float(np.shape(val_ground)[0]))
        f1_all.append(metrics.f1_score(val_ground, predictions))
        pr_all.append(metrics.precision_score(val_ground, predictions))
        re_all.append(metrics.recall_score(val_ground, predictions))

# ### Validation Performance

# In[4]:

ii, jj = np.unravel_index(np.argmax(f1_all), (3, 3))
print('Best Batch Size: ', bs_arr[ii])
print('Best Epochs: ', n_epochs_arr[jj])

print('Validation F1 Score: ', max(f1_all))
print('Validation Best Pr: ', pr_all[np.argmax(f1_all)])
print('Validation Best Re: ', re_all[np.argmax(f1_all)])

# ### Test Performance
# We re-train the model with the best validation performance since we don't save weights for the models currently.

# In[5]:

y_pred = lstm_simple(train_text,
                     train_reef,
                     test_text,
                     test_ground,
                     bs=bs_arr[ii],
                     n=n_epochs_arr[jj])
predictions = np.round(y_pred)

# In[6]:

print('Test F1 Score: ', metrics.f1_score(test_ground, predictions))
print('Test Precision: ', metrics.precision_score(test_ground, predictions))
print('Test Recall: ', metrics.recall_score(test_ground, predictions))

# ## [Optional] Ground Truth Performance
# We can also train the same model with ground truth labels for the train set to see how far Reef labels are from the best possible performance.

# In[8]:

y_pred = lstm_simple(train_text,
                     train_ground,
                     test_text,
                     test_ground,
                     bs=5,
                     n=10)
predictions = np.round(y_pred)

# ### Test Performance

# In[9]:

print('Test F1 Score: ', metrics.f1_score(test_ground, predictions))
print('Test Precision: ', metrics.precision_score(test_ground, predictions))
print('Test Recall: ', metrics.recall_score(test_ground, predictions))
