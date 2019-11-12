from pprint import pprint
from program_synthesis.functions import count_abstains
import numpy as np
from scipy import sparse

from .label_aggregator import LabelAggregator, odds_to_prob
"""
Problem: die Marginals nach dem one vs all approach sind sich für beide Klassen vieeeeeel zu ähnlich, im Vergleich zu den eindeutigen Ergebnissen davor
"""


class MultiLabelAggregator(object):
    """LabelAggregator Object that learns the accuracies for the heuristics. 

    Copied from Snorkel v0.4 NaiveBayes Model with minor changes for simplicity"""
    def __init__(self, n_classes):
        self.w = [None for c in range(n_classes)]
        self.n_classes = n_classes

    # gets as input L_train
    def train(self, X, n_iter=1000, w0=None, rate=0.01, alpha=0.5, mu=1e-6, \
            sample=False, n_samples=100, evidence=None, warm_starts=False, tol=1e-6, verbose=False):
        #  print("X", X)
        #  print("count abstains", count_abstains(X))
        #  exit(-1)
        # create one vs all matrix
        for i in range(self.n_classes):
            one_vs_all_X = self._one_vs_all(
                X, i)  # <- macht das Sinn für multilabel?!
            one_vs_all_label_aggregator = LabelAggregator()
            one_vs_all_label_aggregator.train(one_vs_all_X,
                                              rate=1e-3,
                                              mu=1e-6,
                                              verbose=False)
            self.w[i] = one_vs_all_label_aggregator.w

    def _one_vs_all(self, X, label):
        # input: -1 abstain, 0,1,2,... labels
        # output: -1 other labels, 0 abstain, 1 this label
        X_new = np.full(X.shape, -1)
        X_new[X == -1] = 0
        X_new[X == label] = 1
        return X_new

    def marginals(self, X):

        # x ist L_val -> also -1 abstain, 0 label A, 1 Label B, 2 Label C etc.
        marginals = [None] * self.n_classes
        #  print("w", self.w)
        for i, w in enumerate(self.w):
            # bevor ich X.dot(w) mache muss ich X erst wieder transformieren
            X_new = sparse.csr_matrix(self._one_vs_all(X, i))
            marginals[i] = odds_to_prob(X_new.dot(w))


#            -> they don't add up to 1! is it because of the intference of abstain?
        marginals = np.transpose(marginals)
        return np.array(marginals)
