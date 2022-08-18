import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from AUC import  AUC_intervals
from check_dataset import readDataset

print("Test for dataset heart-disease")
print()
data = readDataset('datasetUCI/heart.csv')
X = data.drop('target', axis =1)
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_train = pd.DataFrame(X_train_scaled)
# X_test_scaled = scaler.fit_transform(X_test)
# X_test = pd.DataFrame(X_test_scaled)

# classifiers - KNN, Decision Tree, Gausian Native Bayes

def KNN_clasifier_predict (n):
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    knn.fit(X_train, Y_train)
    knn_score = knn.predict_proba(X_test)
    return knn_score

def DecisionTree_classifier_predict ():
    tre = DecisionTreeClassifier(max_depth=3)
    tre.fit(X_train, Y_train)
    tree_score = tre.predict_proba(X_test)
    return tree_score

def GaussianNativeBayes_predict ():
    gaussian = GaussianNB()
    gb = gaussian.fit(X_train, Y_train)
    g_score = gb.predict_proba(X_test)
    return g_score

def classifier_coefficient(classifier):
    p = []
    for i in classifier:
         p.append(i[0])
    return p

def classifier_M():

    collection = []

    # k for knn clasifiers

    k_neigh = [1, 3, 5, 10, 15, 20, 30]

    for j in k_neigh:
        knn_k = KNN_clasifier_predict(j)
        KNN_class = classifier_coefficient(knn_k)
        collection.append(KNN_class)

    gb = GaussianNativeBayes_predict()
    tree = DecisionTree_classifier_predict()

    collection.append(classifier_coefficient(gb))
    collection.append(classifier_coefficient(tree))

    if np.mean(collection) > 0.5:
         return "Belongs to the main class"
    else:
        return "Belongs to the subordinate class"

k_neightbors = 5
knn_p = KNN_clasifier_predict(k_neightbors)
tree = DecisionTree_classifier_predict()
gaussian = GaussianNativeBayes_predict()
class_M = classifier_M()

print("AUC KNN (k =",k_neightbors,'):' , AUC_intervals(Y_test, knn_p))
print("AUC Decision Tree: ", AUC_intervals(Y_test, tree))
print("AUC Gausian Native Bayes: ", AUC_intervals(Y_test, gaussian))
print("Classifier M: ", classifier_M())







