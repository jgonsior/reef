import itertools
from pprint import pprint

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from program_synthesis.functions import get_labels_cutoff


class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, X_val, Y_val, n_classes, b=0.5, n_jobs=4):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.X_val = X_val
        self.Y_val = Y_val
        self.p = np.shape(self.X_val)[1]
        self.b = b
        self.n_jobs = n_jobs
        self.n_classes = n_classes

    def generate_feature_combinations(self, cardinality=1):
        """ 
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over 
        """
        primitive_idx = list(range(self.p))
        feature_combinations = []
        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    def fit_function(self, comb, model):
        """ 
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X_val = self.X_val.iloc[:, list(comb)]
        if np.shape(X_val)[0] == 1:
            X_val = X_val.reshape(-1, 1)

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X_val, self.Y_val)

            # caculate val_acc
            Y_pred = dt.apply(self.X_val.iloc[:, list(comb)])
            from sklearn.metrics import accuracy_score
            #  print("va - ", comb, "\t", accuracy_score(self.Y_val, Y_pred))

            return dt

        elif model == 'lr':
            lr = LogisticRegression(multi_class='auto', n_jobs=self.n_jobs)
            lr.fit(X_val, self.Y_val)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=self.n_jobs)
            nn.fit(X_val, self.Y_val)
            return nn

    def generate_heuristics(self, model, max_cardinality=1):
        """ 
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        #have to make a dictionary?? or feature combinations here? or list of arrays?
        # good question!
        feature_combinations_final = []
        heuristics_final = []
        for cardinality in range(1, max_cardinality + 1):
            feature_combinations = self.generate_feature_combinations(
                cardinality)
            heuristics = []
            for _, comb in enumerate(feature_combinations):
                heuristics.append(self.fit_function(comb, model))

            feature_combinations_final.append(feature_combinations)
            heuristics_final.append(heuristics)
        return heuristics_final, feature_combinations_final

    def beta_optimizer(self, marginals, ground):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(self.b, 1, 10)

        highest_probabilities = np.amax(marginals, axis=1)
        most_likely_labels = np.argmax(marginals, axis=1)
        f1 = []

        for beta in beta_params:
            labels_cutoff = get_labels_cutoff(highest_probabilities,
                                              most_likely_labels, beta,
                                              self.n_classes)
            f1score = f1_score(ground, labels_cutoff, average='weighted')
            #  print("be", beta)
            #  print("gr", ground)
            #  print("lc", labels_cutoff)
            #  print("f1", f1score)
            #  print("\n")
            f1.append(f1score)

        f1 = np.nan_to_num(f1)
        #  print("f1s", f1)
        optimal_beta = beta_params[np.argsort(np.array(f1))[-1]]
        #  print("beta", optimal_beta)
        return optimal_beta

    def find_optimal_beta(self, heuristics, X, feat_combos, ground):
        """ 
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """
        beta_opt = []
        for i, hf in enumerate(heuristics):
            marginals = hf.predict_proba(X.iloc[:, list(feat_combos[i])])
            #  print(i, marginals)
            beta_opt.append(self.beta_optimizer(marginals, ground))
        #  exit(-1)
        return beta_opt
