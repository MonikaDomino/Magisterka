import numpy as np
from scipy.stats.mstats import gmean, hmean


def artimetic_mean(classifier_score):

    mean = np.mean(classifier_score)

    return mean

def weighted_average (classifier_score):

    weighted_avg = np.average(classifier_score)

    return weighted_avg

def geometric_mean(classifier_score):

    geometric = gmean(classifier_score)

    return geometric


def harmonic_mean (classifier_score):

    harmonic = hmean(classifier_score)

    return harmonic
