import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import check_dataset
from voting import voting, counter_and_sort, decision_voting_acc, accuracy_voting, sum_acc, decison_voting, auc_roc_voting
from knn import knn_probability
from aggregation_function import artimetic_mean, geometric_mean, weighted_average

X,Y = check_dataset.readDataset_tic_tac_toe()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# standarize

scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# method 1

for i in range(10):
    voting_gb = voting(X_train, Y_train, X_test, GaussianNB())
    voting_lr = voting(X_train, Y_train, X_test, LogisticRegression())
    voting_dt = voting(X_train, Y_train, X_test, DecisionTreeClassifier())


print("Metoda 1")
print()

decison_voting(voting_gb)
decison_voting(voting_lr)
decison_voting(voting_dt)

print()

# method 2

# for decision 0

acc_0 = []

acc_gb_0, acc_gb_1 = accuracy_voting(voting_gb, Y_test)
acc_lr_0, acc_lr_1 = accuracy_voting(voting_lr, Y_test)
acc_dt_0, acc_dt_1 = accuracy_voting(voting_dt, Y_test)

acc_0.append(acc_gb_0)
acc_0.append(acc_dt_0)
acc_0.append(acc_lr_0)

acc_1 = []

acc_1.append(acc_gb_1)
acc_1.append(acc_dt_1)
acc_1.append(acc_lr_1)

sum_1 = sum_acc(acc_1)
sum_0 = sum_acc(acc_0)

mean_a_1 = artimetic_mean(acc_1)
mean_a_0 = artimetic_mean(acc_0)

gmean_a_1 = geometric_mean(acc_1)
gmean_a_0 = geometric_mean(acc_0)

print()

auc_0 = []

auc_gb_0, auc_gb_1 = auc_roc_voting(voting_gb, Y_test)
auc_lr_0, auc_lr_1 = auc_roc_voting(voting_lr, Y_test)
auc_dt_0, auc_dt_1 = auc_roc_voting(voting_dt, Y_test)

auc_0.append(auc_gb_0)
auc_0.append(auc_dt_0)
auc_0.append(auc_lr_0)

auc_1 = []

auc_1.append(auc_gb_1)
auc_1.append(auc_dt_1)
auc_1.append(auc_lr_1)

sum_1_auc = sum_acc(auc_1)
sum_0_auc = sum_acc(auc_0)

mean_a_1_auc = artimetic_mean(auc_1)
mean_a_0_auc = artimetic_mean(auc_0)

gmean_a_1_auc = geometric_mean(auc_1)
gmean_a_0_auc = geometric_mean(auc_0)


print("Metoda 2")
print()

decision_voting_acc(sum_0, sum_1)
decision_voting_acc(mean_a_0, mean_a_1)
decision_voting_acc(gmean_a_0, gmean_a_1)

print()

decision_voting_acc(sum_0_auc, sum_1_auc)
decision_voting_acc(mean_a_0_auc, mean_a_1_auc)
decision_voting_acc(gmean_a_0_auc, gmean_a_1_auc)

print()
# method 3

print("Metoda 3")
print()

for i in range(10):
    knn_0, knn_1 = knn_probability(X_train, Y_train, X_test)


knn_a0 = artimetic_mean(knn_0)
knn_a1 = artimetic_mean(knn_1)

knn_g0 = geometric_mean(knn_0)
knn_g1 = geometric_mean(knn_1)

knn_w0 = weighted_average(knn_0)
knn_w1 = weighted_average(knn_1)

# print(knn_h0, knn_h1)
# print(knn_g0, knn_g1)
# print(knn_a0, knn_a1)

decision_voting_acc(knn_a0, knn_a1)
decision_voting_acc(knn_g0, knn_g1)
decision_voting_acc(knn_w0, knn_w1)












