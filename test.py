import numpy as np
import random

import pandas
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import check_dataset
from aggregation_function import art, sum, geometric_mean
from knn import decisionkNN_for_random_element, knn_pred, knn_probability, knn_all, accuracy_knn, auc_roc_knn,\
    sum_knn_decision, knn_all_random_element
from voting import accuracy_voting, voting_for_one_classifier, voting_for_oC_random_element, votig_classifier, \
    auc_roc_voting


X,Y = check_dataset.readDataset_parkinson()

# method 1
print('Metoda 1')
print()
print('Dla losowego x:')
print("Naiwny klasyfikator Bayes: ", voting_for_oC_random_element(X, Y, GaussianNB()))
print("Regresja logistyczna: ", voting_for_oC_random_element(X, Y, LogisticRegression(solver='lbfgs', max_iter=1000)))
print("Drzewo decyzyjne: ", voting_for_oC_random_element(X, Y, DecisionTreeClassifier()))
print()

print("Dla całego zestawu danych")
voting_gb = voting_for_one_classifier(X,Y, GaussianNB())
print("ACC Naiwny klasyfikator Bayesa: ", round(accuracy_voting(voting_gb, Y), 3))
voting_lr = voting_for_one_classifier(X,Y, LogisticRegression(solver='lbfgs', max_iter=1000))
print("ACC Regresja logistyczna: ", round(accuracy_voting(voting_lr, Y), 3))
voting_dt = voting_for_one_classifier(X, Y, DecisionTreeClassifier(random_state=0))
print("ACC Drzewo decyzyjne: ", round(accuracy_voting(voting_dt, Y), 3))
voting_all = votig_classifier(X, Y)
print("ACC kolekcja: ", round(accuracy_voting(voting_all, Y),3))
print()

print("AUC Naiwny klasyfikator Bayesa: ", round(auc_roc_voting(voting_gb, Y), 3))
print("AUC Regresja logistyczna: ", round(auc_roc_voting(voting_lr, Y), 3))
print("AUC Drzewo decyzyjne: ", round(auc_roc_voting(voting_dt, Y), 3))
print("AUC kolekcja: ", round(auc_roc_voting(voting_all, Y),3))

print()
print("Metoda 2 - użycie agregacji takich jak suma, średnia arytmetyczna i średnia geometryczna")
print("Średnia arytmetyczna: ")


print("ACC", round(accuracy_voting(art(voting_all)>0.5, Y),3))
print("AUC", round(auc_roc_voting(art(voting_all)>0.5, Y), 3))

print()
print("Suma: ")
print("ACC", round(accuracy_voting(sum(voting_all)>0.5, Y),3))
print("AUC", round(auc_roc_voting(sum(voting_all)>0.5, Y), 3))

print()
print("Średnia geometryczna: ")
print("ACC", round(accuracy_voting(geometric_mean(voting_all), Y),3))
print("AUC", round(auc_roc_voting(geometric_mean(voting_all), Y), 3))

# print("Metoda 3:")
# print()
# print("Dla losowego elementu dla danego k")
# print("k = 3")
# decisionkNN_for_random_element(3, X_train, Y_train, X_test)
# print("k = 5")
# decisionkNN_for_random_element(5, X_train, Y_train, X_test)
# print("k = 7")
# decisionkNN_for_random_element(7, X_train, Y_train, X_test)
# print("k = 15")
# decisionkNN_for_random_element(15, X_train, Y_train, X_test)
# print("k = 20")
# decisionkNN_for_random_element(20, X_train, Y_train, X_test)
# print("k = 30")
# decisionkNN_for_random_element(30, X_train, Y_train, X_test)
# print()
# knn_3 = knn_pred(3, X_train, Y_train, X_test)
# knn_5 = knn_pred(5, X_train, Y_train, X_test)
# knn_7 = knn_pred(7, X_train, Y_train, X_test)
# knn_15 = knn_pred(15, X_train, Y_train, X_test)
# knn_20 = knn_pred(20, X_train, Y_train, X_test)
# knn_30 = knn_pred(30, X_train, Y_train, X_test)
# print()
# print("ACC dla róznych k")
# print(round(accuracy_voting(knn_3, Y_test),3))
# print(round(accuracy_voting(knn_5, Y_test),3))
# print(round(accuracy_voting(knn_7, Y_test),3))
# print(round(accuracy_voting(knn_15, Y_test),3))
# print(round(accuracy_voting(knn_20, Y_test),3))
# print(round(accuracy_voting(knn_30, Y_test),3))
#
# print()
# print("AUC dla róznych k")
# print(round(auc_roc_voting(knn_3, Y_test),3))
# print(round(auc_roc_voting(knn_5, Y_test),3))
# print(round(auc_roc_voting(knn_7, Y_test),3))
# print(round(auc_roc_voting(knn_15, Y_test),3))
# print(round(auc_roc_voting(knn_20, Y_test),3))
# print(round(auc_roc_voting(knn_30, Y_test),3))
# print()
# k = knn_all(X_train,Y_train, X_test)
#
# print("ACC dla kolekcji klasyfikatorów kNN ")
# # # accuracy for all knn
# print(round(accuracy_knn(k, Y_test), 3))
# print("AUC dla kolekcji klasyfikatorów kNN ")
# # # auc for all knn
# print(round(auc_roc_knn(k, Y_test), 3))
# print()
# sum_knn_decision(X_train, Y_train, X_test)
# artimetic_knn_decision(X_train, Y_train, X_test)
# t_norm_knn(X_train, Y_train, X_test)
# t_konorm_knn(X_train, Y_train, X_test)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
































