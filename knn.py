import random

from sklearn.neighbors import KNeighborsClassifier
from aggregation_function import artimetic_mean, geometric_mean

from voting import counter_and_sort


def knn_pred(k, x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test[:k])
    return predicted


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

    # print("probability decision 1: ", prob_1)
    # print("probability decision 0: ", prob_0)


# knn probability
def knn_probability(x_train, y_train, x_test):

    prob_knn_0 = []
    prob_knn_1 = []

    k_neigh = [3, 5, 7, 15, 20,  30]

    for i in k_neigh:
        knn_i = knn_pred(i, x_train, y_train, x_test)
        p_0, p_1 = probability_decision(knn_i)
        prob_knn_0.append(p_0)
        prob_knn_1.append(p_1)

    return prob_knn_0, prob_knn_1













