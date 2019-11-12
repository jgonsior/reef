from pprint import pprint

import numpy as np


def get_labels_cutoff(highest_probabilities, most_likely_labels, beta,
                      n_classes):
    labels_cutoff = np.zeros(np.shape(highest_probabilities))
    for i, (highest_probability, most_likely_label) in enumerate(
            zip(highest_probabilities, most_likely_labels)):
        abstain_adjusted_probability = highest_probability - (beta / n_classes)
        if highest_probability == 1 / n_classes:
            # abstain in case the probabilities beforehand were the same
            #  print("hui")
            labels_cutoff[i] = -1
        elif abstain_adjusted_probability > beta:
            labels_cutoff[i] = most_likely_label
        else:
            labels_cutoff[i] = -1
            #  print("hui")

    return labels_cutoff


def marginals_to_labels(hf, X, beta, n_classes, debug=False):
    marginals = hf.predict_proba(X)
    highest_probabilities = np.amax(marginals, axis=1)
    most_likely_labels = np.argmax(marginals, axis=1)
    marginals = get_labels_cutoff(highest_probabilities, most_likely_labels,
                                  beta, n_classes)
    #  if count_abstains(marginals) != len(marginals):
    #  if debug:
    #  print("X", X)
    #  print("marg", marginals)
    #  print("bet", beta)
    #  print("high prob", highest_probabilities)
    #  print("most likely", most_likely_labels)
    #  print("cutoff", marginals)
    #  print("\n" * 2)
    return marginals


def count_abstains(X):
    return np.count_nonzero(np.where(X == -1))
