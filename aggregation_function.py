import numpy as np
from scipy.stats.mstats import gmean


def artimetic_mean(classifier_score):

    mean = np.mean(classifier_score)

    return mean

def weighted_average (classifier_score):

    weighted_avg = np.average(classifier_score)

    return weighted_avg

def geometric_mean(classifier_score):

    geometric = gmean(classifier_score)

    return geometric


def sum_acc(acc_array):
    sum = np.sum(acc_array)
    return sum

def t_norm_Lukasiewicz(array_acc):
    sum_array_tn = np.sum(array_acc)
    sum_1 = sum_array_tn-1
    max_tnorm = max(sum_1, 0)

    return max_tnorm


def t_konorm_Lukasiewicz(array_acc):
    sum_array_tn = np.sum(array_acc)
    min_tkonorm = min(1, sum_array_tn)
    return min_tkonorm
