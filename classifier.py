import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from boostrap_AUC import boostrap_ACC

# read data from file
#data = pd.read_csv('datasetUCI/heart.csv')
path = 'datasetUCI/breast-cancer-wisconsin.data'
names = ['Id', 'Clump_thickness','Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', 'Single_e_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(path, names=names)
print(data.info())
print(data.Class.value_counts())

X = data.drop('Class', axis =1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
scaler = MinMaxScaler()
print(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

# classifiers - KNN, Decision Tree, Gausian Native Bayes

def KNN_clasifier_predict (n, x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, Y_train)
    #knn_score = knn.predict_proba(X_test)
    return knn

def DecisionTree_classifier_predict (x_train, y_train):
    tre = DecisionTreeClassifier()
    tre.fit(x_train, y_train)
    #tree_score = tre.predict_proba(x_test)
    return tre

def GaussianNativeBayes (X_train, Y_train):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    #g_score = gaussian.predict_proba(X_test)
    return gaussian

knn = KNN_clasifier_predict(15, X_train, Y_train)
intervals_AUC_KNN = boostrap_ACC(X_train, Y_train, knn)
print("AUC KNN: ", intervals_AUC_KNN)

tree = DecisionTree_classifier_predict(X_train, Y_train)
intervals_AUC_Tree = boostrap_ACC(X_train, Y_train, tree)
print("AUC Decision Tree: ", intervals_AUC_Tree)

gausian = GaussianNativeBayes(X_train, Y_train)
intervals_AUC_Gausian = boostrap_ACC(X_train, Y_train, gausian)
print("AUC Native Bayes: ", intervals_AUC_Gausian)
