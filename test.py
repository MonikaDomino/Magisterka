
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import check_dataset
from knn import knn_pred_random_element, knn_pred_acc, knn_pred_auc, knn_for_art_acc, knn_for_art_auc, \
    knn_for_art_random_element, knn_for_gmean_random_element, knn_for_gmean_acc, knn_for_gmean_auc,\
    knn_for_hmean_random_element, knn_for_hmean_acc, knn_for_hmean_auc
from voting import voting_for_one_classifier_acc, voting_for_oC_random_element, votig_classifier_harMean_acc, \
    votig_classifier_harm_auc, votig_classifier_all_acc, votig_classifier_all_auc, voting_for_one_classifier_auc, \
    votig_classifier_artimetic_mean_acc, votig_classifier_artimetic_mean_auc, votig_classifier_geometric_mean_acc, \
    votig_classifier_geometric_mean_auc, voting_for_all_random_element_agregate_sum, \
    voting_for_all_random_element_agregate_art,voting_for_all_random_element_agregate_geometric


X,Y = check_dataset.readDataset_breastCancer()

#method 1
# print('Metoda 1')
# print()
# print('Dla losowego x:')
# print("Naiwny klasyfikator Bayes: ", voting_for_oC_random_element(X, Y, GaussianNB()))
# print("Regresja logistyczna: ", voting_for_oC_random_element(X, Y, LogisticRegression(solver='lbfgs', max_iter=10000)))
# print("Drzewo decyzyjne: ", voting_for_oC_random_element(X, Y, DecisionTreeClassifier()))
# print()
#
# print("Dla całego zestawu danych")
# voting_gb_acc = voting_for_one_classifier_acc(X,Y, GaussianNB())
# print("ACC Naiwny klasyfikator Bayesa: ", round(voting_gb_acc, 3))
# voting_lr_acc = voting_for_one_classifier_acc(X,Y, LogisticRegression(solver='lbfgs', max_iter=10000))
# print("ACC Regresja logistyczna: ", round(voting_lr_acc,3))
# voting_dt_acc = voting_for_one_classifier_acc(X, Y, DecisionTreeClassifier(random_state=0))
# print("ACC Drzewo decyzyjne: ", round(voting_dt_acc, 3))
#
#
# print("ACC kolekcja: ", round(votig_classifier_all_acc(X,Y),3))
# print()
# print("AUC Naiwny klasyfikator Bayesa: ", round(voting_for_one_classifier_auc(X, Y, GaussianNB()), 3))
# print("AUC Regresja logistyczna: ", round(voting_for_one_classifier_auc(X, Y, LogisticRegression(solver='lbfgs', max_iter=10000)), 3))
# print("AUC Drzewo decyzyjne: ", round(voting_for_one_classifier_auc(X, Y, DecisionTreeClassifier(random_state=0)), 3))
# print("AUC kolekcja: ", round(votig_classifier_all_auc(X, Y),3))

# print()
# print("Metoda 2 - użycie agregacji takich jak suma, średnia arytmetyczna i średnia geometryczna")
# print()
# print('Dla losowego x:')
# print("Średnia arytmetyczna: ", voting_for_all_random_element_agregate_art(X,Y))
# print("Średnia harmoniczna: ", voting_for_all_random_element_agregate_sum(X,Y))
# print("Średnia geometryczna: ", voting_for_all_random_element_agregate_geometric(X,Y))
# print()
#
#
# print("Średnia arytmetyczna: ")
# print("ACC", round(votig_classifier_artimetic_mean_acc(X,Y),3))
# print("AUC", round(votig_classifier_artimetic_mean_auc(X,Y), 3))

# print("Średnia harmoniczna:")
# print("ACC:", votig_classifier_harMean_acc(X, Y))
# print("AUC:", votig_classifier_harm_auc(X, Y))

# print()
# print("Średnia geometryczna: ")
# print("ACC", round(votig_classifier_geometric_mean_acc(X,Y), 3))
# print("AUC", round(votig_classifier_geometric_mean_auc(X,Y), 3))
# print()
print("Metoda 3:")
print()
print("Dla losowego elementu dla danego k")
k_neight = [3,5,7,15,20,30]

for i in k_neight:
    print("k = ", i, ': ', knn_pred_random_element(i, X, Y))

print()

print("ACC dla róznych k")

k_neight = [3,5,7,15,20,30]

for i in k_neight:
    print("k = ", i, ': ', knn_pred_acc(i, X, Y))

print()
print("AUC dla róznych k")

for i in k_neight:
    print("k = ", i, ': ', knn_pred_auc(i, X, Y))

print()
print("Średnia arytmetyczna - random element")

for i in k_neight:
    print("k = ", i, ':', knn_for_art_random_element(X, Y, i))

print()
print("Średnia arytmetyczna ACC:")

for j in k_neight:
    print("k = ", j, ':', knn_for_art_acc(X, Y, j))

print()
print("Średnia arytmetyczna AUC:")

for j in k_neight:
    print("k = ", j, ':', knn_for_art_auc(X, Y, j))

print()
print("Średnia harmoniczna - random elem")

for i in k_neight:
    print("k = ", i, ':', knn_for_hmean_random_element(X, Y, i))

print()
print("Średnia harmoniczna - ACC")

for j in k_neight:
    print("k = ", j, ':', knn_for_hmean_acc(X,Y,j))

print()
print("Średnia harmoniczna - AUC")

for j in k_neight:
    print("k = ", j, ':', knn_for_hmean_auc(X, Y, j))

print()
print("Średnia geometryczna - random element")

for k in k_neight:
    print("k = ", k, ':', knn_for_gmean_random_element(X,Y,k))

print()
print("Średnia geometryczna - ACC")

for k in k_neight:
    print("k = ", k, ':', knn_for_gmean_acc(X, Y, k))

print()
print("Średnia geometryczna - AUC")

for k in k_neight:
    print("k = ", k, ':', knn_for_gmean_auc(X, Y, k))





































