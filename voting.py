import random

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import aggregation_function

def accuracy_voting(voting, Y):
    acc_score = accuracy_score(Y, voting)
    return acc_score

def auc_roc_voting(voting, Y):
    auc_score = roc_auc_score(Y, voting)
    return auc_score

def voting_classifier_all():
    vote = VotingClassifier(estimators=[('gnb', GaussianNB()), ('lr', LogisticRegression(solver='lbfgs', max_iter=10000)),
                                        ('dt', DecisionTreeClassifier())], voting='hard')
    return vote

def voting_classifier_one(classifier):
    vote_one = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')
    return vote_one

def votig_classifier_all_acc(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    vote = voting_classifier_all()
    scores_acc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        scores_acc.append(accuracy_voting(Y_test, vote_predict))

    return round(np.mean(scores_acc), 3)

def votig_classifier_all_auc(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    vote = voting_classifier_all()
    scores_auc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        scores_auc.append(auc_roc_voting(Y_test, vote_predict))

    return round(np.mean(scores_auc), 3)

def votig_classifier_harMean_acc(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    vote = voting_classifier_all()
    scores_acc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        sum_acc = aggregation_function.harmonic_mean(vote_predict)
        scores_acc.append(accuracy_voting(Y_test, sum_acc))

    return round(np.mean(scores_acc), 3)

def votig_classifier_harm_auc(X, Y):

    vote_auc = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote_auc.fit(X_train, Y_train)
        vote_predict = vote_auc.predict(X_test)
        sum_acc = aggregation_function.harmonic_mean(vote_predict)
        scores.append(auc_roc_voting(Y_test, sum_acc))

    return round(np.mean(scores), 3)

def votig_classifier_artimetic_mean_acc(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    vote = voting_classifier_all()
    scores_acc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        sum_acc = aggregation_function.art(vote_predict)
        scores_acc.append(accuracy_voting(Y_test, sum_acc))

    return round(np.mean(scores_acc), 3)

def votig_classifier_artimetic_mean_auc(X, Y):

    vote_auc = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote_auc.fit(X_train, Y_train)
        vote_predict = vote_auc.predict(X_test)
        sum_acc = aggregation_function.art(vote_predict)
        scores.append(auc_roc_voting(Y_test, sum_acc))

    return round(np.mean(scores), 3)


def votig_classifier_geometric_mean_acc(X, Y):

    kf = KFold(n_splits=10, shuffle=True)
    vote = voting_classifier_all()
    scores_acc = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        sum_acc = aggregation_function.geometric_mean(vote_predict)
        scores_acc.append(accuracy_voting(Y_test, sum_acc))

    return round(np.mean(scores_acc), 3)

def votig_classifier_geometric_mean_auc(X, Y):

    vote_auc = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote_auc.fit(X_train, Y_train)
        vote_predict = vote_auc.predict(X_test)
        sum_acc = aggregation_function.geometric_mean(vote_predict)
        scores.append(auc_roc_voting(Y_test, sum_acc))

    return round(np.mean(scores), 3)

def voting_for_one_classifier_acc(X, Y, classifier):
    vote = voting_classifier_one(classifier)
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        scores.append(accuracy_voting(Y_test,vote_predict))

    return np.mean(scores)

def voting_for_one_classifier_auc(X, Y, classifier):
    vote = voting_classifier_one(classifier)
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        scores.append(auc_roc_voting(Y_test,vote_predict))

    return np.mean(scores)

def voting_for_oC_random_element (X, Y, classifier):
    vote = voting_classifier_one(classifier)
    kf = KFold(n_splits=10, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        scores.append(vote_predict)

    k_number = random.randint(0, len(scores))
    ar = np.array(scores, dtype=object)
    return random.choice(ar[k_number-1])

def voting_for_all_random_element_agregate_sum (X, Y):
    vote = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    scores_sum = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        sum_acc = aggregation_function.harmonic_mean(vote_predict)

        scores_sum.append(sum_acc)
        k_number = random.randint(0, len(scores_sum))
        ar = np.array(scores_sum)

        return random.choice(ar[k_number-1])

def voting_for_all_random_element_agregate_art(X, Y):
    vote = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    score_art = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        art = aggregation_function.art(vote_predict)
        score_art.append(art)

        k_number = random.randint(0, len(score_art))
        ar = np.array(score_art)

        return random.choice(ar[k_number-1])

def voting_for_all_random_element_agregate_geometric (X, Y):
    vote = voting_classifier_all()
    kf = KFold(n_splits=10, shuffle=True)
    score_art = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        vote.fit(X_train, Y_train)
        vote_predict = vote.predict(X_test)
        art = aggregation_function.geometric_mean(vote_predict)
        score_art.append(art)

        k_number = random.randint(0, len(score_art))
        ar = np.array(score_art)

        return random.choice(ar[k_number-1])

def counter_and_sort(array_voting):
    count_one = []
    count_zero = []

    for i in array_voting:

        if i == 0:
            count_zero.append(i)
        else:
            count_one.append(i)

    return count_zero, count_one


def decision_voting(math_function_acc_0, math_function_acc_1):
    if math_function_acc_0 > math_function_acc_1:
        print("Decision is 0")
    else:
        print("Decision is 1")






