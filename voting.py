import random
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def votig_classifier(X, Y):
    vote = VotingClassifier(estimators=[('gnb', GaussianNB()), ('lr', LogisticRegression(solver='lbfgs', max_iter=1000)),
                                        ('dt', DecisionTreeClassifier())], voting='hard')

    scores = cross_val_predict(vote, X, Y, cv = 10)

    return scores


def voting_for_one_classifier(X, Y, classifier):
    vote = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')
    scores = cross_val_predict(vote, X, Y, cv=10)

    return scores

def voting_for_oC_random_element (X, Y, classifier):
    vote = VotingClassifier(estimators=[('classifier', classifier)], voting='hard')
    scores = cross_val_predict(vote, X, Y, cv=10)
    k = random.randint(0, len(X))

    return scores[k].reshape(1, -1)


def accuracy_voting(voting, Y):
    acc_score = accuracy_score(voting, Y)
    return acc_score

def auc_roc_voting(voting, Y):
    auc_score = roc_auc_score(Y, voting)
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






