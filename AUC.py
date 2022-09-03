import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, roc_auc_score


def calculate_Z_score(alfa):
    Z_score = norm.ppf((1 + (1 - alfa)) / 2.0)
    Z_score = abs(round(Z_score, 3))
    return Z_score


def AUC_intervals(y_test, score):
    rounds = 10
    auc_array = []
    se_auc = []

    for i in range(rounds):
        # fpr, tpr, threshold = roc_curve(y_test, score)
        # roc_auc = auc(fpr, tpr)tpr
        roc_auc = roc_auc_score(y_test, score, multi_class='ovr')
        auc_array.append(roc_auc)

        alfa = 0.05
        z_value = calculate_Z_score(alfa)
        se = z_value * np.sqrt((roc_auc * (1 - roc_auc)) / y_test.shape[0])
        se_auc.append(se)

    auc_roc = np.mean(auc_array)
    se_roc = np.mean(se_auc)

    auc_roc = round(auc_roc, 3)
    se_roc = round(se_roc, 3)

    return [auc_roc, '+/-', se_roc]
