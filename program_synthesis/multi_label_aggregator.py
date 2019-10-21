from pprint import pprint

import numpy as np
from scipy import sparse

from .label_aggregator import LabelAggregator, odds_to_prob


class MultiLabelAggregator(object):
    """LabelAggregator Object that learns the accuracies for the heuristics. 

    Copied from Snorkel v0.4 NaiveBayes Model with minor changes for simplicity"""
    def __init__(self, n_classes):
        self.w = [None for c in range(n_classes)]
        self.n_classes = n_classes

    def train(self, X, n_iter=1000, w0=None, rate=0.01, alpha=0.5, mu=1e-6, \
            sample=False, n_samples=100, evidence=None, warm_starts=False, tol=1e-6, verbose=False):
        # create one vs all matrix
        for i in range(self.n_classes):
            one_vs_all_X = self._one_vs_all(X, i)
            one_vs_all_label_aggregator = LabelAggregator()
            one_vs_all_label_aggregator.train(one_vs_all_X,
                                              rate=1e-3,
                                              mu=1e-6,
                                              verbose=False)
            self.w[i] = one_vs_all_label_aggregator.w

    def marginals(self, X):
        #  print("w")
        #  pprint(self.w)
        marginals = [None] * self.n_classes
        for i, w in enumerate(self.w):
            # bevor ich X.dot(w) mache muss ich X erst wieder transformieren
            X_new = sparse.csr_matrix(self._one_vs_all(X, i))
            marginals[i] = odds_to_prob(X_new.dot(w))
        marginals = np.transpose(marginals)
        #  print("neue marginals")
        #  print(marginals)
        #  pprint(np.array(marginals).shape)
        return np.array(marginals)

    def _one_vs_all(self, X, label):
        """ Create a one vs all encoded matrix """
        #  print("davor: ", label)
        #  pprint(X)
        X_new = np.zeros(X.shape)
        X_new[X == -1] = -1
        X_new[X == label] = 1

        #  print("danach:")
        #  pprint(X_new)
        #  print("\n")
        return X_new
