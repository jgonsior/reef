from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from program_synthesis.functions import get_labels_cutoff, marginals_to_labels
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier


class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """
    def __init__(self,
                 X_train,
                 X_val,
                 Y_val,
                 Y_train=None,
                 n_classes=2,
                 n_jobs=4,
                 b=0.5):
        """ 
        Initialize HeuristicGenerator object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """

        self.X_train = X_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.Y_train = Y_train
        self.b = b
        self.vf = None
        self.syn = None
        self.hf = []
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        self.feat_combos = []

    def apply_heuristics(self,
                         heuristics,
                         primitive_matrix,
                         feat_combos,
                         beta_opt,
                         debug=False):
        """ 
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """
        L = np.zeros((np.shape(primitive_matrix)[0], len(heuristics)))
        for i, hf in enumerate(heuristics):
            #  if debug:
            #  print(i, ": \t",
            #  primitive_matrix.iloc[:, list(feat_combos[i])])
            L[:, i] = marginals_to_labels(
                hf,
                primitive_matrix.iloc[:, list(feat_combos[i])],
                beta_opt[i],
                self.n_classes,
                debug=debug)
        return L

    def prune_heuristics(self, heuristics, feat_combos, keep=1):
        """ 
        Selects the best heuristic based on Jaccard Distance and Reliability Metric

        keep: number of heuristics to keep from all generated heuristics
        """

        #  -> check based on example downwards if this calculates the right thing
        #  -> the trick is, that sometimes the sum of abs(-1, 1) is being taken

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(
                    np.minimum(
                        num_labeled_L[:, i], num_labeled_total)) / np.sum(
                            np.maximum(num_labeled_L[:, i], num_labeled_total))
            return 1 - scores

        L_vals = np.array([])
        L_trains = np.array([])
        beta_opts = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i],
                                                       self.X_val,
                                                       feat_combos[i],
                                                       self.Y_val)
            #  print(i, "beta ", beta_opt_temp)
            L_val_temp = self.apply_heuristics(heuristics[i],
                                               self.X_val,
                                               feat_combos[i],
                                               beta_opt_temp,
                                               debug=False)
            L_train_temp = self.apply_heuristics(heuristics[i],
                                                 self.X_train,
                                                 feat_combos[i],
                                                 beta_opt_temp,
                                                 debug=False)
            #  print(beta_opt_temp)
            beta_opts = np.append(beta_opts, beta_opt_temp)
            if i == 0:
                L_vals = np.append(
                    L_vals, L_val_temp)  #converts to 1D array automatically
                L_vals = np.reshape(L_vals, np.shape(L_val_temp))
                L_trains = np.append(
                    L_trains,
                    L_train_temp)  #converts to 1D array automatically
                L_trains = np.reshape(L_trains, np.shape(L_train_temp))
            else:
                pprint("UIUIUIU" * 10000)
                L_vals = np.concatenate((L_vals, L_val_temp), axis=1)
                L_trains = np.concatenate((L_trains, L_train_temp), axis=1)
        #  print("L_val", L_vals)
        #Use F1 trade-off for reliability
        acc_cov_scores = [
            f1_score(
                self.Y_val,
                L_vals[:, i],
                average='micro',
            ) for i in range(np.shape(L_vals)[1])
        ]
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        #  -> vc berechnung stimmt ni -> dann nach und nach den imdb_small datensatz größer machen bis die Ergebnisse ni mehr miteinander übereinstimmen
        #  print("\n" * 5)
        #  for i in range(np.shape(L_vals)[1]):
        #  print(i, L_vals[:, i])
        #  print("acc_cov_scores", np.sort(acc_cov_scores))
        #  print("\n" * 5)

        if self.vf != None:
            #Calculate Jaccard score for diversity
            #  @todo stimmt das hier?!
            #  lieber die formeln von unten für accuracy und coverage nehmen?!

            #  Es sieht so aus als ob die accuracie gleich bleiben, wohingegen die coverages immer größer werden
            train_num_labeled = np.sum(self.vf.L_train >= 0, axis=1)
            jaccard_scores = calculate_jaccard_distance(
                train_num_labeled, np.abs(L_trains))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))
        #  print("accs", acc_cov_scores)
        #  print("jaccs", jaccard_scores)
        #Weighting the two scores to find best heuristic
        combined_scores = 0.5 * acc_cov_scores + 0.5 * jaccard_scores
        sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        return sort_idx

    def run_synthesizer(self, max_cardinality=1, idx=None, keep=1, model='lr'):
        """ 
        Generates Synthesizer object and saves all generated heuristics

        max_cardinality: max number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        model: train logistic regression ('lr') or decision tree ('dt')
        """
        if idx == None:
            # first run, use the whole dataset
            X_val = self.X_val
            Y_val = self.Y_val
        else:
            # only use the points from the validation dataset for finding heuristics which had low confidence before!
            X_val = self.X_val.iloc[idx, :]
            Y_val = np.array(self.Y_val)[idx]

        #Generate all possible heuristics
        self.syn = Synthesizer(X_val,
                               Y_val,
                               n_classes=self.n_classes,
                               b=self.b,
                               n_jobs=self.n_jobs)

        #Un-flatten indices
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i += 1
            try:
                return a[i - 1][
                    remainder]  #TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb
                pdb.set_trace()

        #Select keep best heuristics from generated heuristics
        hf, feat_combos = self.syn.generate_heuristics(model, max_cardinality)
        sort_idx = self.prune_heuristics(hf, feat_combos, keep)
        for i in sort_idx:
            self.hf.append(index(hf, i))
            self.feat_combos.append(index(feat_combos, i))
        #create appended L matrices for validation and train set
        beta_opt = self.syn.find_optimal_beta(self.hf, self.X_val,
                                              self.feat_combos, self.Y_val)
        self.L_val = self.apply_heuristics(self.hf,
                                           self.X_val,
                                           self.feat_combos,
                                           beta_opt,
                                           debug=False)
        self.L_train = self.apply_heuristics(self.hf, self.X_train,
                                             self.feat_combos, beta_opt)

    def run_verifier(self):
        """ 
        Generates Verifier object and saves marginals
        """
        self.vf = Verifier(self.L_train,
                           self.L_val,
                           self.Y_val,
                           self.n_classes,
                           has_snorkel=False)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def gamma_optimizer(self, marginals):
        """ 
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        """
        m = len(self.hf)
        gamma = 0.5 - (1 / (m**(3 / 2.)))
        return gamma

    def find_feedback(self):
        """ 
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        #TODO: flag for re-classifying incorrect points
        #incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        #gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        #  incorrect_idx = vague_idx

        # @todo: no concatenation but union!
        #  self.feedback_idx = list(set(list(np.concatenate(
        #  (vague_idx)))))  #, incorrect_idx)))))
        self.feedback_idx = list(vague_idx)

    def calculate_accuracy(self, marginals, b, Y_true):
        #  hier werden marginals jetzt ohne b berechnet? bzw. nur die als abstain gezählt, die auch tatsächlich exackt b sind?
        #  wie sieht das in der feedback bestimmung aus?

        Y_pred = np.argmax(marginals, axis=1)
        # abstain for labels where the prediction isn't clear
        #  print("marginals", marginals)
        indices_with_abstain = np.where(np.amax(marginals, axis=1) == b)
        #  print("indic_w_abst", list(indices_with_abstain))
        for i in indices_with_abstain[0]:
            #  print(i)
            #  if len(i) == 0:
            #  continue
            i = int(i)
            Y_pred[i] = Y_true[i]
        return accuracy_score(Y_true, Y_pred)

    def calculate_coverage(self, marginals, b, Y_true):
        #  print("marg", marginals)
        #  print("b", b)

        highest_probabilities = np.amax(marginals, axis=1)
        Y_test = [-1 if prob == 0.5 else 1 for prob in highest_probabilities]
        amount_of_labels_not_abstain = len(Y_test)
        #  print("amount_of_labels_not_abstain", amount_of_labels_not_abstain)
        total_labels = np.shape(Y_test)[0]
        #  print("total_labels", total_labels)
        #  print("\n")
        return amount_of_labels_not_abstain / total_labels

    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """

        # why? :crying_emoji:
        self.val_marginals = self.vf.val_marginals
        self.train_marginals = self.vf.train_marginals

        self.val_accuracy = self.calculate_accuracy(self.val_marginals, self.b,
                                                    self.Y_val)
        self.train_accuracy = self.calculate_accuracy(self.train_marginals,
                                                      self.b, self.Y_train)
        self.val_coverage = self.calculate_coverage(self.val_marginals, self.b,
                                                    self.Y_val)
        self.train_coverage = self.calculate_coverage(self.train_marginals,
                                                      self.b, self.Y_train)

        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage

    def heuristic_stats(self):
        '''For each heuristic, we want the following:
        - idx of the features it relies on
        - if dt, then the thresholds?
        '''

        stats_table = np.zeros((len(self.hf), 6))
        for i in range(len(self.hf)):
            stats_table[i, 0] = int(self.feat_combos[i][0])
            try:
                stats_table[i, 1] = int(self.feat_combos[i][1])
            except:
                stats_table[i, 1] = -1.
            stats_table[i, 2] = self.calculate_accuracy(
                self.L_val[:, i], self.b, self.Y_val)
            stats_table[i, 3] = self.calculate_accuracy(
                self.L_train[:, i], self.b, self.Y_train)
            stats_table[i, 4] = self.calculate_coverage(
                self.L_val[:, i], self.b, self.Y_val)
            stats_table[i, 5] = self.calculate_coverage(
                self.L_train[:, i], self.b, self.Y_train)

        #Make table
        column_headers = [
            'Feat 1', 'Feat 2', 'Val Acc', 'Train Acc', 'Val Cov', 'Train Cov'
        ]
        pandas_stats_table = pd.DataFrame(stats_table, columns=column_headers)
        return pandas_stats_table
