
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import check_dataset
from aggregation_function import art, sum, geometric_mean
from knn import decisionkNN_for_random_element, knn_pred, knn_probability, knn_all, accuracy_knn, auc_roc_knn,\
    sum_knn_decision, knn_all_random_element
from voting import voting_for_one_classifier_acc, voting_for_oC_random_element, votig_classifier_sum_acc, \
    votig_classifier_sum_auc, votig_classifier_all_acc, votig_classifier_all_auc, voting_for_one_classifier_auc, \
    votig_classifier_artimetic_mean_acc, votig_classifier_artimetic_mean_auc, votig_classifier_geometric_mean_acc, \
    votig_classifier_geometric_mean_auc, voting_for_all_random_element_agregate_sum, \
    voting_for_all_random_element_agregate_art,voting_for_all_random_element_agregate_geometric


X,Y = check_dataset.readDataset_diabets()

# method 1
print('Metoda 1')
print()
print('Dla losowego x:')
print("Naiwny klasyfikator Bayes: ", voting_for_oC_random_element(X, Y, GaussianNB()))
print("Regresja logistyczna: ", voting_for_oC_random_element(X, Y, LogisticRegression(solver='lbfgs', max_iter=10000)))
print("Drzewo decyzyjne: ", voting_for_oC_random_element(X, Y, DecisionTreeClassifier()))
print()
#
print("Dla całego zestawu danych")
voting_gb_acc = voting_for_one_classifier_acc(X,Y, GaussianNB())
print("ACC Naiwny klasyfikator Bayesa: ", round(voting_gb_acc, 3))
voting_lr_acc = voting_for_one_classifier_acc(X,Y, LogisticRegression(solver='lbfgs', max_iter=10000))
print("ACC Regresja logistyczna: ", round(voting_lr_acc,3))
voting_dt_acc = voting_for_one_classifier_acc(X, Y, DecisionTreeClassifier(random_state=0))
print("ACC Drzewo decyzyjne: ", round(voting_dt_acc, 3))


print("ACC kolekcja: ", round(votig_classifier_all_acc(X,Y),3))
print()
print("AUC Naiwny klasyfikator Bayesa: ", round(voting_for_one_classifier_auc(X, Y, GaussianNB()), 3))
print("AUC Regresja logistyczna: ", round(voting_for_one_classifier_auc(X, Y, LogisticRegression(solver='lbfgs', max_iter=10000)), 3))
print("AUC Drzewo decyzyjne: ", round(voting_for_one_classifier_auc(X, Y, DecisionTreeClassifier(random_state=0)), 3))
0,print("AUC kolekcja: ", round(votig_classifier_all_auc(X, Y),3))
#
print()
print("Metoda 2 - użycie agregacji takich jak suma, średnia arytmetyczna i średnia geometryczna")
# k = 1
#
print('Dla losowego x:')
print("Suma: ", voting_for_all_random_element_agregate_sum(X,Y))
print("Średnia arytmetyczna: ", voting_for_all_random_element_agregate_art(X,Y))
print("Średnia geometryczna: ", voting_for_all_random_element_agregate_geometric(X,Y))
print()


print("Średnia arytmetyczna: ")
print("ACC", round(votig_classifier_artimetic_mean_acc(X,Y),3))
print("AUC", round(votig_classifier_artimetic_mean_auc(X,Y), 3))

print()
print("Suma:")
print("ACC:", votig_classifier_sum_acc(X, Y))
print("AUC:", votig_classifier_sum_auc(X, Y))

print()
print("Średnia geometryczna: ")
print("ACC", round(votig_classifier_geometric_mean_acc(X,Y), 3))
print("AUC", round(votig_classifier_geometric_mean_auc(X,Y), 3))

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
































