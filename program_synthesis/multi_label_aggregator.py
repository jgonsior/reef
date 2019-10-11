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
        print("w")
        pprint(self.w)
        marginals = [None] * self.n_classes
        for i, w in enumerate(self.w):
            marginals[i] = odds_to_prob(X.dot(w))
        marginals = np.transpose(marginals)
        print("neue marginals")
        print(marginals)
        pprint(np.array(marginals).shape)
        return np.array(marginals)

    def _one_vs_all(self, X, label):
        """ Create a one vs all encoded matrix """
        X_new = X.copy()
        X_new[np.logical_and(X_new != label, X_new != -1)] = -5
        X_new[X_new == label] = 1

        #  exchange -1 for abstain and 0
        X_new[X_new == -1] = 0
        X_new[X_new == -5] = -1

        return X_new
