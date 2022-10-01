import random
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve
from aggregation_function import t_konorm_Lukasiewicz, t_norm_Lukasiewicz, artimetic_mean, geometric_mean, \
    weighted_average, sum
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def votig_classifier(X_train, Y_train, X_test):
    vote = VotingClassifier(estimators=[('gnb', GaussianNB()),('lr', LogisticRegression()),
                                        ('dt', DecisionTreeClassifier()) ], voting='hard')
    vote.fit(X_train, Y_train)
    vote_predict = vote.predict(X_test)
    return vote_predict

def voting_random_element_classifiers(X_train, Y_train, X_test):
    vote = VotingClassifier(estimators=[('gnb', GaussianNB()),('lr', LogisticRegression()),
                                        ('dt', DecisionTreeClassifier()) ], voting='hard')
    vote.fit(X_train, Y_train)
    k = random.randint(0, len(X_test))
    predicted = vote.predict(X_test[k].reshape(1, -1))
    return predicted


def voting_for_one_classifier(X_train, Y_train, X_test, classifier):
    vote = VotingClassifier(estimators=[('classifier',classifier )], voting='hard')

    vote.fit(X_train, Y_train)
    vote_predict = vote.predict(X_test)

    return vote_predict

def voting_random_element_one_classifier(X_train, Y_train, X_test, classifier):
    vote = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')
    vote.fit(X_train, Y_train)
    k = random.randint(0, len(X_test))
    predicted = vote.predict(X_test[k].reshape(1, -1))
    return predicted


def accuracy_voting(voting, y_test):
    acc_score = accuracy_score(voting, y_test)
    return acc_score

def auc_roc_voting(voting, y_test):
    auc_score = roc_auc_score(y_test, voting)
    return auc_score

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

def sum_voting_decision(voting_array):
    a0, a1 = counter_and_sort(voting_array)
    sum_0 = sum(a0)
    sum_1 = sum(a1)

    decision_voting(sum_0, sum_1)

def artimetic_mean_voting_decison(voting_array):
    a0, a1 = counter_and_sort(voting_array)
    art_m_0 = artimetic_mean(a0)
    art_m_1 = artimetic_mean(a1)

    decision_voting(art_m_0, art_m_1)

def t_norm_voting_decision (voting_array):
    t0, t1 = counter_and_sort(voting_array)
    tn_0 = t_norm_Lukasiewicz(t0)
    tn_1 = t_norm_Lukasiewicz(t1)

    decision_voting(tn_0, tn_1)

def t_konorm_voting_decision (voting_array):
    tk0, tk1 = counter_and_sort(voting_array)
    tk_0 = t_konorm_Lukasiewicz(tk0)
    tk_1 = t_konorm_Lukasiewicz(tk1)

    decision_voting(tk_0, tk_1)





