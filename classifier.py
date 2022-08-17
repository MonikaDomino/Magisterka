import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from boostrap_AUC import boostrap_ACC

# read data from file
data = pd.read_csv('datasetUCI/heart.csv')
#path = 'datasetUCI/breast-cancer-wisconsin.data'
#names = ['Id', 'Clump_thickness','Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', 'Single_e_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
#data = pd.read_csv(path, names=names)
#print(data.info())
# print(data.Class.value_counts())

X = data.drop('target', axis =1)
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

# classifiers - KNN, Decision Tree, Gausian Native Bayes

def KNN_clasifier_predict (n, x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, Y_train)
    knn_score = knn.predict_proba(x_test)
    return knn_score

def DecisionTree_classifier_predict (x_train, y_train, x_test):
    tre = DecisionTreeClassifier()
    tre.fit(x_train, y_train)
    tree_score = tre.predict_proba(x_test)
    return tree_score

def GaussianNativeBayes_predict (X_train, Y_train, x_test):
    gaussian = GaussianNB()
    gb = gaussian.fit(X_train, Y_train)
    g_score = gb.predict_proba(x_test)
    return g_score


gb = GaussianNativeBayes_predict(X_train, Y_train, X_test)
tree = DecisionTree_classifier_predict(X_train, Y_train, X_test)
knn = KNN_clasifier_predict(10, X_train, Y_train, X_test)

def classifier_M (classifier):
    p = []

    for i in classifier:
         p.append(i[0])

    if np.mean(p) > 0.5:
        return "Belongs to the main class"
    else:
        return "Belongs to the subordinate class"

print("Classifier M for Gaussian Native Bayes: ", classifier_M(gb))
print("Classifier M for Decision Tree: ", classifier_M(tree))
print("Classifier M for KNN: ", classifier_M(knn))

