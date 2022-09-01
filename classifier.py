from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets


from AUC import  AUC_intervals

from check_dataset import readDataset_plik, readDataset_url
from collection_classifier import voting_classifier_hard, stacking_classifier, voting_classifier_soft

print("Test for dataset heart-disease")
print()
data = readDataset_plik()
X = data.drop('target', axis=1)
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

#create MinMaxScaler object
scaler_min_max = MinMaxScaler()

#fit object to data
scaler_min_max.fit(X_train)

#get transformed train data
X_train = scaler_min_max.transform(X_train)

#get transformed test data
X_test = scaler_min_max.transform(X_test)


# classifiers - KNN, Decision Tree, Gausian Native Bayes

def KNN_clasifier_predict(k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, Y_train)
    knn_score = knn.predict(X_test)
    return knn_score


def DecisionTree_classifier_predict():
    tre = DecisionTreeClassifier(max_depth=3)
    tre.fit(X_train, Y_train)
    tree_score = tre.predict(X_test)

    return tree_score


def GaussianNativeBayes_predict():
    gaussian = GaussianNB()
    gb = gaussian.fit(X_train, Y_train)
    g_score = gb.predict(X_test)
    return g_score


def auc_classifier_knn():
    k_neigh = [1, 3, 5, 10, 15, 20, 30]

    for j in k_neigh:
        knn_k = KNN_clasifier_predict(j)
        print("AUC KNN (k =", j, '):', AUC_intervals(Y_test, knn_k))


tree = DecisionTree_classifier_predict()
gaussian = GaussianNativeBayes_predict()

votCH = voting_classifier_hard(X_train, Y_train, X_test)
votS = voting_classifier_soft(X_train, Y_train, X_test)


sC = stacking_classifier(X_train, Y_train, X_test)

auc_classifier_knn()
print()
print("AUC Decision Tree: ", AUC_intervals(Y_test, tree))
print("AUC Gausian Native Bayes: ", AUC_intervals(Y_test, tree))
print()
print("AUC Voting Classier hard: ", AUC_intervals(Y_test, votCH))
print("AUC Voting Classier soft: ", AUC_intervals(Y_test,votS))
print("AUC Stacking Classifier: ", AUC_intervals(Y_test,sC))
