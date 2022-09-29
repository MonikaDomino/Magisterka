import random
import numpy as np
from sklearn.ensemble import VotingClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve
from aggregation_function import sum_acc, t_konorm_Lukasiewicz, t_norm_Lukasiewicz, artimetic_mean, geometric_mean, \
    weighted_average


def voting(X_train, Y_train, X_test, classifier):
    vote = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')

    vote.fit(X_train, Y_train)
    vote_predict = vote.predict(X_test)

    return vote_predict


# zliczanie ilość wystąpień 0 i 1 w zestawie

def counter(array_voting):
    return Counter(array_voting)


def counter_and_sort(array_voting):
    count_one = []
    count_zero = []

    for i in array_voting:

        if i == 0:
            count_zero.append(i)
        else:
            count_one.append(i)

    return count_zero, count_one


def voting_random_element(X_train, Y_train, X_test, classifier):
    vote = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')
    vote.fit(X_train, Y_train)
    k = random.randint(0, len(X_test))
    predicted = vote.predict(X_test[k].reshape(1, -1))
    return predicted



def accuracy_voting(voting, y_test):
    acc_dec_0, acc_dec_1 = counter_and_sort(voting)
    acc_dec_len_0 = len(acc_dec_0)
    acc_dec_len_1 = len(acc_dec_1)

    acc_score_0 = accuracy_score(acc_dec_0, y_test.iloc[:acc_dec_len_0])
    acc_score_1 = accuracy_score(acc_dec_1, y_test.iloc[:acc_dec_len_1])

    return acc_score_0, acc_score_1


def auc_roc_voting(voting, y_test):
    auc_roc_dec_0, auc_roc_dec_1 = counter_and_sort(voting)
    auc_roc_dec_len_0 = len(auc_roc_dec_0)
    auc_roc_dec_len_1 = len(auc_roc_dec_1)

    auc_score_0 = roc_auc_score(y_test.iloc[:auc_roc_dec_len_0], auc_roc_dec_0)
    auc_score_1 = roc_auc_score(y_test.iloc[:auc_roc_dec_len_1], auc_roc_dec_1)

    return auc_score_0, auc_score_1


def decision_voting(math_function_acc_0, math_function_acc_1):
    if math_function_acc_0 > math_function_acc_1:
        print("Decision is 0")
    else:
        print("Decision is 1")


