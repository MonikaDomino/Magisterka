import random
import numpy as np
import pandas

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

import aggregation_function
from voting import counter_and_sort, decision_voting


def knn_pred_random_element(k, X, Y):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn.fit(X_train, Y_train)
        vote_predict = knn.predict(X_test[:k])
        scores.append(vote_predict)

    k_number = random.randint(0, len(scores))
    ar = np.array(scores, dtype=object)
    return random.choice(ar[k_number - 1])


def knn_pred_acc(k, X, Y):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn.fit(X_train, Y_train)
        vote_predict = knn.predict(X_test)
        scores.append(accuracy_knn(vote_predict, Y_test))

    return round(np.mean(scores),3)


def knn_pred_auc(k, X, Y):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn.fit(X_train, Y_train)
        vote_predict = knn.predict(X_test)
        scores.append(auc_roc_knn(vote_predict, Y_test))

    return round(np.mean(scores), 3)


# probability decisions
def probability_decision_for_one(array_knn):
    #  print(counter(array_knn))

    X_0, Y_1 = counter_and_sort(array_knn)

    numbers_0 = len(X_0)
    numbers_1 = len(Y_1)

    all_decision = len(array_knn)

    prob_1 = numbers_1 / all_decision
    prob_0 = numbers_0 / all_decision

    if (prob_1 > prob_0):
        print('Decision is 1')
    else:
        print('Decision is 0')


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


def accuracy_knn(array_knn, y_test):
    score_acc = accuracy_score(array_knn, y_test)
    return score_acc


def auc_roc_knn(array_knn, y_test):
    auc_score = roc_auc_score(y_test, array_knn)
    return auc_score

def knn_for_art_random_element(X, Y, k):
    k_acc_art = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.art(knn_predict_art)
        k_acc_art.append(art_knn)

        k_number = random.randint(0, len(k_acc_art))
        ar = np.array(k_acc_art)

        return random.choice(ar[k_number - 1])

def knn_for_art_acc(X, Y, k):
    k_acc_art = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.art(knn_predict_art)
        k_acc_art.append(accuracy_knn(art_knn, Y_test))


        return round(np.mean(k_acc_art),3)


def knn_for_art_auc(X, Y, k):
    k_auc_art = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.art(knn_predict_art)
        k_auc_art.append(auc_roc_knn(art_knn, Y_test))


        return round(np.mean(k_auc_art),3)

def knn_for_hmean_random_element(X, Y, k):
    k_acc_hmean = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.harmonic_mean(knn_predict_art)
        k_acc_hmean.append(art_knn)

        k_number = random.randint(0, len(k_acc_hmean))
        ar = np.array(k_acc_hmean)

        return random.choice(ar[k_number - 1])

def knn_for_hmean_acc(X, Y, k):
    k_hmean_acc = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.harmonic_mean(knn_predict_art)
        k_hmean_acc.append(accuracy_knn(art_knn, Y_test))


        return round(np.mean(k_hmean_acc),3)


def knn_for_hmean_auc(X, Y, k):
    k_auc_hmean = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.harmonic_mean(knn_predict_art)
        k_auc_hmean.append(auc_roc_knn(art_knn, Y_test))


        return round(np.mean(k_auc_hmean),3)


def knn_for_gmean_random_element(X, Y, k):
    k_acc_gmean = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.geometric_mean(knn_predict_art)
        k_acc_gmean.append(art_knn)

        k_number = random.randint(0, len(k_acc_gmean))
        ar = np.array(k_acc_gmean)

        return random.choice(ar[k_number - 1])

def knn_for_gmean_acc(X, Y, k):
    k_gmean_acc = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.geometric_mean(knn_predict_art)
        k_gmean_acc.append(accuracy_knn(art_knn, Y_test))


        return round(np.mean(k_gmean_acc),3)


def knn_for_gmean_auc(X, Y, k):
    k_auc_gmean = []

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, Y_train)
        knn_predict_art = knn.predict(X_test)
        art_knn = aggregation_function.geometric_mean(knn_predict_art)
        k_auc_gmean.append(auc_roc_knn(art_knn, Y_test))


        return round(np.mean(k_auc_gmean),3)