from pprint import pprint

import numpy as np
from scipy import sparse

from .label_aggregator import LabelAggregator
from .multi_label_aggregator import MultiLabelAggregator


def odds_to_prob(l):
    """
  This is the inverse logit function logit^{-1}:
    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
    return np.exp(l) / (1.0 + np.exp(l))


class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """
    def __init__(self, X_train, X_val, Y_val, n_classes, has_snorkel=True):
        self.X_train = X_train.astype(int)
        self.X_val = X_val.astype(int)
        self.Y_val = Y_val
        self.n_classes = n_classes

    def train_gen_model(self, deps=False, grid_search=False):
        """ 
        Calls appropriate generative model
        """
        gen_model = MultiLabelAggregator(self.n_classes)
        gen_model.train(self.X_train, rate=1e-3, mu=1e-6, verbose=True)
        #  marginals = gen_model.marginals(self.X_train)
        #  for x, marginal in zip(self.X_train, marginals):
        #  print(x, "\t -> \t", np.argmax(marginal), "\t", marginal)
        self.gen_model = gen_model

    def assign_marginals(self):
        """ 
        Assigns probabilistic labels for train and val sets 
        """
        self.train_marginals = self.gen_model.marginals(self.X_train)
        self.val_marginals = self.gen_model.marginals(self.X_val)
        #  for marginal in self.val_marginals:
        #  print(marginal)
        #print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self, gamma=0.1, b=0.5):
        """ 
        Find val set indices where marginals are within thresh of b 
        # returns the first point of the validation set which is as close as gamma to the marginal
        """
        # PROBLEM: probabilities sind vieeeeeeeel zu groß im Vergleich zu denen von den binären Labels
        #  print("gamma:", gamma)
        #  print("b:", b)
        #  print("val_marginals", self.val_marginals)
        result = []
        for i, marginal in enumerate(self.val_marginals):
            max_prob = np.amax(marginal)
            if max_prob - b / self.n_classes <= gamma:
                result.append(i)
        #  print("val_idx", val_idx)
        #  exit(-1)
        return result

    def find_incorrect_points(self, b=0.5):
        print("find_incorrect_points klappt ni")
        """ Find val set indices where marginals are incorrect """
        Y_val = 2 * (self.val_marginals > b) - 1
        val_idx = np.where(Y_val != self.Y_val)
        return val_idx[0]
