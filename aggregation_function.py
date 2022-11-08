import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean, hmean

def art (classifier_score):
    df_clas = pd.DataFrame(classifier_score)
    df_art = df_clas.rolling(1).agg(np.mean)
    return df_art


def harmonic_mean (classifier_score):
    df_clas = pd.DataFrame(classifier_score)
    df_sum = df_clas.rolling(1).apply(hmean)
    return df_sum


def geometric_mean (classifier_score):

    df_max = pd.DataFrame(classifier_score)
    df_t_min = df_max.rolling(1).apply(gmean)
    return df_t_min

