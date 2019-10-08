import pprint
import numpy as np


def get_labels_cutoff(highest_probabilities, most_likely_labels, beta):
    labels_cutoff = np.zeros(np.shape(highest_probabilities))

    for i, (highest_probability, most_likely_label) in enumerate(
            zip(highest_probabilities, most_likely_labels)):
        if highest_probability > beta:
            labels_cutoff[i] = most_likely_label
        else:
            # -1 means abstain as 0 is a normal label!
            labels_cutoff[i] = -1
    return labels_cutoff


def marginals_to_labels(hf, X, beta):
    marginals = hf.predict_proba(X)
    highest_probabilities = np.amax(marginals, axis=1)
    most_likely_labels = np.argmax(marginals, axis=1)

    return get_labels_cutoff(highest_probabilities, most_likely_labels, beta)
