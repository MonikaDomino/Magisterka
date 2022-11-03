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
        vote_predict = knn.predict(X_test)
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

    return np.mean(scores)


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

    return np.mean(scores)


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


def knn_all_acc(X, Y):
    k_scores = []
    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)
            k_scores.append(accuracy_knn(predict, Y_test))

    return np.mean(k_scores)


def knn_all_auc(X, Y):
    k_scores = []
    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)
            k_scores.append(auc_roc_knn(predict, Y_test))

    return np.mean(k_scores)


def knn_for_all_random_element_agregate(X, Y):
    k_scores_sum = []
    k_scores_art = []
    k_scores_gmean = []
    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)
            sum = aggregation_function.sum(predict)
            k_scores_sum.append(sum)

            art = aggregation_function.art(predict)
            k_scores_art.append(art)

            gm = aggregation_function.geometric_mean(predict)
            k_scores_gmean.append(gm)

        k_number_sum = random.randint(0, len(k_scores_sum))
        s = np.array(k_scores_sum)
        random_elem_sum = random.choice(s[k_number_sum - 1])

        k_number_art = random.randint(0, len(k_scores_art))
        a = np.array(k_scores_art)
        random_elem_art = random.choice(a[k_number_art - 1])

        k_number_gmean = random.randint(0, len(k_scores_gmean))
        g = np.array(k_scores_gmean)
        random_elem_gmean = random.choice(g[k_number_gmean - 1])

        return random_elem_sum, random_elem_art, random_elem_gmean


def knn_for_all_sum_acc(X, Y):
    k_acc_sum = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            sum = aggregation_function.sum(predict)
            k_acc_sum.append(accuracy_knn(sum, Y_test))

        sum_acc = np.mean(k_acc_sum)

        return sum_acc


def knn_for_all_art_acc(X, Y):
    k_acc_art = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            art = aggregation_function.art(predict)
            k_acc_art.append(accuracy_knn(art, Y_test))

        return np.mean(k_acc_art)


def knn_for_all_gmean_acc(X, Y):
    k_acc_gmean = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            art = aggregation_function.geometric_mean(predict)
            k_acc_gmean.append(accuracy_knn(art, Y_test))

        return np.mean(k_acc_gmean)

def knn_for_all_sum_auc(X, Y):
    k_acc_sum = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            sum = aggregation_function.sum(predict)
            k_acc_sum.append(auc_roc_knn(sum, Y_test))

        sum_acc = np.mean(k_acc_sum)

        return sum_acc


def knn_for_all_art_auc(X, Y):
    k_acc_art = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            art = aggregation_function.art(predict)
            k_acc_art.append(auc_roc_knn(art, Y_test))

        return np.mean(k_acc_art)


def knn_for_all_gmean_auc(X, Y):
    k_acc_gmean = []

    kf = KFold(n_splits=10, shuffle=True)
    k_neighbours = [3, 5, 7, 15, 20, 30]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        for k in k_neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            predict = knn.predict(X_test)

            art = aggregation_function.geometric_mean(predict)
            k_acc_gmean.append(auc_roc_knn(art, Y_test))

        return np.mean(k_acc_gmean)