import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import check_dataset
from knn import knn_probability
from voting import voting, accuracy_voting, auc_roc_voting, voting_random_element, decision_voting
from aggregation_function import artimetic_mean, sum_acc,t_norm_Lukasiewicz, t_konorm_Lukasiewicz

X,Y = check_dataset.readDataset_parkinson()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# standarize

scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# method 1

print("Naiwny klasyfikator Bayes: ", voting_random_element(X_train,Y_train, X_test, GaussianNB()))
print("Regresja logistyczna ", voting_random_element(X_train, Y_train, X_test, LogisticRegression()))
print("Drzewo decyzyjne ", voting_random_element(X_train, Y_train, X_test, DecisionTreeClassifier()))

# method 2

# voting_gb = voting(X_train, Y_train, X_test, GaussianNB())
# voting_lr = voting(X_train, Y_train, X_test, LogisticRegression())
# voting_dt = voting(X_train, Y_train, X_test, DecisionTreeClassifier())
#
# # ACC decyzji 0 i 1
#
# accGB_0, acc_GB_1 = accuracy_voting(voting_gb, Y_test)
# acc_lr_0, acc_lr_1 = accuracy_voting(voting_lr, Y_test)
# acc_dt_0, acc_dt_1 = accuracy_voting(voting_dt, Y_test)
#
# acc_0 = []
# acc_0.append(accGB_0)
# acc_0.append(acc_lr_0)
# acc_0.append(acc_dt_0)
#
# acc_1 = []
# acc_1.append(acc_GB_1)
# acc_1.append(acc_lr_1)
# acc_1.append(acc_dt_1)


# print()
# print("ACC - Naiwny klasyfikator Bayesa dla decyzji 0: ", round(accGB_0,3))
# print("ACC - Regresja logistyczna dla decyzji 0: ", round(acc_lr_0, 3))
# print("ACC - Drzewo klasyfikacyjne dla decyzji 0:", round(acc_dt_0, 3))
# print()
# print("ACC - Naiwny klasyfikator Bayesa dla decyzji 1: ", round(acc_GB_1,3))
# print("ACC - Regresja logistyczna dla decyzji 1: ", round(acc_lr_1, 3))
# print("ACC - Drzewo klasyfikacyjne dla decyzji 1:", round(acc_dt_1, 3))
#
#
# sumACC0 = sum_acc(acc_0)
# sumACC1 = sum_acc(acc_1)
#
# am_ACC0 = artimetic_mean(acc_0)
# am_ACC1 = artimetic_mean(acc_1)
#
# tn_0 = t_norm_Lukasiewicz(acc_0)
# tn_1 = t_norm_Lukasiewicz(acc_1)
#
# tkn_0 = t_konorm_Lukasiewicz(acc_0)
# tkn_1 = t_konorm_Lukasiewicz(acc_1)
#
# print()
# decision_voting(sumACC0, sumACC1)
# decision_voting(am_ACC0, am_ACC1)
# decision_voting(tn_0, tn_1)
# decision_voting(tkn_0, tkn_1)
#
#
# aucGB_0, auc_GB_1 = auc_roc_voting(voting_gb, Y_test)
# auc_lr_0, auc_lr_1 = auc_roc_voting(voting_lr, Y_test)
# auc_dt_0, auc_dt_1 = auc_roc_voting(voting_dt, Y_test)
#
# auc_0 = []
# auc_0.append(aucGB_0)
# auc_0.append(auc_lr_0)
# auc_0.append(auc_dt_0)
#
# auc_1 = []
# auc_1.append(auc_GB_1)
# auc_1.append(auc_lr_1)
# auc_1.append(auc_dt_1)
#
# print()
# print("AUC - Naiwny klasyfikator Bayesa dla decyzji 0: ", aucGB_0)
# print("AUC - Regresja logistyczna dla decyzji 0: ", round(auc_lr_0, 3))
# print("AUC - Drzewo klasyfikacyjne dla decyzji 0:", round(auc_dt_0, 3))
# print()
# print("AUC - Naiwny klasyfikator Bayesa dla decyzji 1: ", round(auc_GB_1,3))
# print("AUC - Regresja logistyczna dla decyzji 1: ", round(auc_lr_1, 3))
# print("AUC - Drzewo klasyfikacyjne dla decyzji 1:", round(auc_dt_1, 3))
#
#
# sumAUC0 = sum_acc(auc_0)
# sumAUC1 = sum_acc(auc_1)
#
# am_AUC0 = artimetic_mean(auc_0)
# am_AUC1 = artimetic_mean(auc_1)
#
# tn_0_AUC = t_norm_Lukasiewicz(auc_0)
# tn_1_AUC = t_norm_Lukasiewicz(auc_1)
#
# tkn_0_AUC = t_konorm_Lukasiewicz(auc_0)
# tkn_1_AUC = t_konorm_Lukasiewicz(auc_1)
#
# print()
# decision_voting(sumAUC0, sumAUC1)
# decision_voting(am_AUC0, am_AUC1)
# decision_voting(tn_0_AUC, tn_1_AUC)
# decision_voting(tkn_0_AUC, tkn_1_AUC)

# method 3

print()

k_0, k_1 = knn_probability(X_train, Y_train, X_test)
print('Prawdopodobieństwa decyzji 0:')
print(np.round(k_0,3))
print('Prawdopodobieństwa decyzji 1:')
print(np.round(k_1, 3)  )

sum_knn0 = sum_acc(k_0)
sum_knn1 = sum_acc(k_1)

am_knn0 = artimetic_mean(k_0)
am_knn1 = artimetic_mean(k_1)

tn_0_knn = t_norm_Lukasiewicz(k_0)
tn_1_knn = t_norm_Lukasiewicz(k_1)

tkn_0_knn = t_konorm_Lukasiewicz(k_0)
tkn_1_knn = t_konorm_Lukasiewicz(k_1)

print()
decision_voting(sum_knn0, sum_knn1)
decision_voting(am_knn0, am_knn1)
decision_voting(tn_0_knn, tn_1_knn)
decision_voting(tkn_0_knn, tkn_1_knn)














