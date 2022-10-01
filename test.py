import numpy as np
import random

import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import check_dataset
from aggregation_function import sum,t_norm_Lukasiewicz, t_konorm_Lukasiewicz, artimetic_mean
from knn import decisionkNN_for_random_element, knn_pred, knn_probability, knn_all, accuracy_knn, auc_roc_knn,\
    sum_knn_decision, artimetic_knn_decision, t_norm_knn, t_konorm_knn, knn_all_random_element
from voting import accuracy_voting,voting_for_one_classifier, voting_random_element_one_classifier, auc_roc_voting, \
    sum_voting_decision,artimetic_mean_voting_decison, t_norm_voting_decision, t_konorm_voting_decision, voting_random_element_classifiers, votig_classifier



X,Y = check_dataset.readDataset_parkinson()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# standarize

scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# method 1
print('Metoda 1')
print()
print('Dla losowego x:')
print("Naiwny klasyfikator Bayes: ", voting_random_element_one_classifier(X_train,Y_train, X_test, GaussianNB()))
print("Regresja logistyczna ", voting_random_element_one_classifier(X_train, Y_train, X_test, LogisticRegression()))
print("Drzewo decyzyjne ", voting_random_element_one_classifier(X_train, Y_train, X_test, DecisionTreeClassifier()))
print()

print("Dla całego zestawu danych")
voting_gb = voting_for_one_classifier(X_train, Y_train, X_test, GaussianNB())
acc_gb = accuracy_voting(voting_gb, Y_test)
print("Naiwny klasyfikator Bayesa_ACC: ", round(acc_gb, 3))
voting_lr = voting_for_one_classifier(X_train, Y_train, X_test, LogisticRegression())
acc_lr = accuracy_voting(voting_lr, Y_test)
print("Regresja logistyczna_ACC: ", round(acc_lr, 3))
voting_dt = voting_for_one_classifier(X_train, Y_train, X_test, DecisionTreeClassifier())
acc_dt = accuracy_voting(voting_dt, Y_test)
print("Drzewo decyzyjne_ACC: ", round(acc_dt, 3))
print()

auc_gb = auc_roc_voting(voting_gb, Y_test)
print("Naiwny klasyfikator Bayesa AUC: ", round(auc_gb, 3))
voting_lr = voting_for_one_classifier(X_train, Y_train, X_test, LogisticRegression())
auc_lr = auc_roc_voting(voting_lr, Y_test)
print("Regresja logistyczna AUC: ", round(auc_lr, 3))
voting_dt = voting_for_one_classifier(X_train, Y_train, X_test, DecisionTreeClassifier())
auc_dt = auc_roc_voting(voting_dt, Y_test)
print("Drzewo decyzyjne AUC: ", round(auc_dt, 3))
print()

# naiwny klasyfikator Bayesa
print('Ostateczne decyzje - Naiwny klasyfikator Bayesa:')
print('Suma:')
sum_voting_decision(voting_gb)
print('Średnia arytmetyczna:')
artimetic_mean_voting_decison(voting_gb)
print('T-norma Lukasiewicza:')
t_norm_voting_decision(voting_gb)
print('T-konorma Lukasiewicza:')
t_konorm_voting_decision(voting_gb)
print()

print('Ostateczne decyzje - Regresja logistyczna:')
print('Suma:')
sum_voting_decision(voting_lr)
print('Średnia arytmetyczna:')
artimetic_mean_voting_decison(voting_lr)
print('T-norma Lukasiewicza:')
t_norm_voting_decision(voting_lr)
print('T-konorma Lukasiewicza:')
t_konorm_voting_decision(voting_lr)
print()

print('Ostateczne decyzje - Drzewo decyzyjne:')
print('Suma:')
sum_voting_decision(voting_dt)
print('Średnia arytmetyczna:')
artimetic_mean_voting_decison(voting_dt)
print('T-norma Lukasiewicza:')
t_norm_voting_decision(voting_dt)
print('T-konorma Lukasiewicza:')
t_konorm_voting_decision(voting_dt)

print()
print("Metoda 2:")
#method 2
print("Decyzja dla losowego obiektu:")
print(voting_random_element_classifiers(X_train, Y_train, X_test))
print()
print("Efektywność ACC i AUC: ")
acc_2 = accuracy_voting(votig_classifier(X_train, Y_train, X_test), Y_test)
print(round(acc_2,3))
auc_2 = auc_roc_voting(votig_classifier(X_train, Y_train, X_test), Y_test)
print(round(auc_2, 3))
print()


print('Suma:')
sum_voting_decision(votig_classifier(X_train, Y_train, X_test))
print('Średnia arytmetyczna:')
artimetic_mean_voting_decison(votig_classifier(X_train, Y_train, X_test))
print('T-norma Lukasiewicza:')
t_norm_voting_decision(votig_classifier(X_train, Y_train, X_test))
print('T-konorma Lukasiewicza:')
t_konorm_voting_decision(votig_classifier(X_train, Y_train, X_test))
print()

print("Metoda 3:")
print()
print("Dla losowego elementu dla danego k")
print("k = 3")
decisionkNN_for_random_element(3, X_train, Y_train, X_test)
print("k = 5")
decisionkNN_for_random_element(5, X_train, Y_train, X_test)
print("k = 7")
decisionkNN_for_random_element(7, X_train, Y_train, X_test)
print("k = 15")
decisionkNN_for_random_element(15, X_train, Y_train, X_test)
print("k = 20")
decisionkNN_for_random_element(20, X_train, Y_train, X_test)
print("k = 30")
decisionkNN_for_random_element(30, X_train, Y_train, X_test)
print()
knn_3 = knn_pred(3, X_train, Y_train, X_test)
knn_5 = knn_pred(5, X_train, Y_train, X_test)
knn_7 = knn_pred(7, X_train, Y_train, X_test)
knn_15 = knn_pred(15, X_train, Y_train, X_test)
knn_20 = knn_pred(20, X_train, Y_train, X_test)
knn_30 = knn_pred(30, X_train, Y_train, X_test)
print()
print("ACC dla róznych k")
print(round(accuracy_voting(knn_3, Y_test),3))
print(round(accuracy_voting(knn_5, Y_test),3))
print(round(accuracy_voting(knn_7, Y_test),3))
print(round(accuracy_voting(knn_15, Y_test),3))
print(round(accuracy_voting(knn_20, Y_test),3))
print(round(accuracy_voting(knn_30, Y_test),3))

print()
print("AUC dla róznych k")
print(round(auc_roc_voting(knn_3, Y_test),3))
print(round(auc_roc_voting(knn_5, Y_test),3))
print(round(auc_roc_voting(knn_7, Y_test),3))
print(round(auc_roc_voting(knn_15, Y_test),3))
print(round(auc_roc_voting(knn_20, Y_test),3))
print(round(auc_roc_voting(knn_30, Y_test),3))
print()
k = knn_all(X_train,Y_train, X_test)

print("ACC dla kolekcji klasyfikatorów kNN ")
# # accuracy for all knn
print(round(accuracy_knn(k, Y_test), 3))
print("AUC dla kolekcji klasyfikatorów kNN ")
# # auc for all knn
print(round(auc_roc_knn(k, Y_test), 3))
print()
sum_knn_decision(X_train, Y_train, X_test)
artimetic_knn_decision(X_train, Y_train, X_test)
t_norm_knn(X_train, Y_train, X_test)
t_konorm_knn(X_train, Y_train, X_test)



























































