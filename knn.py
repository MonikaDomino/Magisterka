import random
import numpy as np
import pandas

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from voting import counter_and_sort,decision_voting



def knn_pred_random_element(k, x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test[:k])
    return predicted

def knn_pred(k, x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test)
    return predicted


# probability decisions
def probability_decision_for_one(array_knn):
    #  print(counter(array_knn))

    X_0, Y_1 = counter_and_sort(array_knn)

    numbers_0 = len(X_0)
    numbers_1 = len(Y_1)

    all_decision = len(array_knn)

    prob_1 = numbers_1 / all_decision
    prob_0 = numbers_0 / all_decision

    if(prob_1 > prob_0):
        print('Decision is 1')
    else:
        print('Decision is 0')



def decisionkNN_for_random_element(k, x_train, y_train, x_test):
    array_k = knn_pred_random_element(k, x_train, y_train, x_test)
    decision = probability_decision_for_one(array_k)
    return decision




# probability decisions
def probability_decision(array_knn):
    #  print(counter(array_knn))

    X_0, Y_1 = counter_and_sort(array_knn)

    numbers_0 = len(X_0)
    numbers_1 = len(Y_1)

    all_decision = len(array_knn)

    prob_1 = numbers_1 / all_decision
    prob_0 = numbers_0 / all_decision

    return prob_0, prob_1

# knn probability
def knn_probability(x_train, y_train, x_test):

    prob_knn_0 = []
    prob_knn_1 = []

    k_neigh = [3, 5, 7, 15, 20,  30]
    knn_all = []

    for i in k_neigh:
        knn_i = knn_pred(i, x_train, y_train, x_test)
        knn_all.append(knn_i)
        p_0, p_1 = probability_decision(knn_i)
        prob_knn_0.append(p_0)
        prob_knn_1.append(p_1)


    return prob_knn_0, prob_knn_1


def knn_all (x_train, y_train, x_test):
    knnA = []
    k_neigh = [3, 5, 7, 15, 20, 30]

    for i in k_neigh:
        knn_i = knn_pred(i, x_train, y_train, x_test)
        for j in knn_i:
            knnA.append(j)

    knnA = pandas.DataFrame(knnA)
    return knnA

def knn_all_random_element(x_train, y_train, x_test):
    knnA_r = []
    k_neigh = [3, 5, 7, 15, 20, 30]
    for i in k_neigh:
        knn_i = knn_pred_random_element(i, x_train, y_train, x_test)
        for j in knn_i:
            knnA_r.append(j)

    return knnA_r



def accuracy_knn(array_knn, y_test):
    score_acc = accuracy_score(array_knn.iloc[:len(y_test)], y_test)
    return score_acc

def sum_knn_decision(X_train, Y_train, X_test):
    p0, p1 = knn_probability(X_train, Y_train, X_test)
    sum_0 = sum(p0)
    sum_1 = sum(p1)
    decision_voting(sum_0, sum_1)

# def artimetic_knn_decision(X_train, Y_train, X_test):
#     p0, p1 = knn_probability(X_train, Y_train, X_test)
#     a_0 = artimetic_mean(p0)
#     a_1 = artimetic_mean(p1)
#     decision_voting(a_0, a_1)
# #
# def t_norm_knn(X_train, Y_train, X_test):
#     p0, p1 = knn_probability(X_train, Y_train, X_test)
#     tn_0 = t_norm_Lukasiewicz(p0)
#     tn_1 = t_norm_Lukasiewicz(p1)
#     decision_voting(tn_0, tn_1)
#
# def t_konorm_knn(X_train, Y_train, X_test):
#     p0, p1 = knn_probability(X_train, Y_train, X_test)
#     tn_0 = t_norm_Lukasiewicz(p0)
#     tn_1 = t_norm_Lukasiewicz(p1)
#     decision_voting(tn_0, tn_1)


def auc_roc_knn(array_knn, y_test):
    auc_score = roc_auc_score(y_test, array_knn.iloc[:len(y_test)])
    return auc_score









