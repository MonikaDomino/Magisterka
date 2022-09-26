import pandas as pd
import numpy as np
import openpyxl

from sklearn.preprocessing import LabelEncoder

# Breast Cancer Wisconsin - class [benign_0__mal_1]
def readDataset_breastCancer():
    print("Test for dataset Breast Cancer Wisconsin - Diagnosic")
    print()
    data = pd.read_csv("datasetUCI/cancer_classification.csv")
    #print(data.info())
    print(data['benign_0__mal_1'].value_counts())
    X = data.drop(columns=['benign_0__mal_1'], axis=1)
    Y = data['benign_0__mal_1']

    return X, Y


# heart disease - class [target]
def readDataset_heart():
    print("Test for dataset heart-disease")
    print()
    data = pd.read_csv("datasetUCI/heart.csv")
    # print(data.info())
    X = data.drop(columns=['target'], axis=1)
    Y = data['target']

    return X, Y

# red wine quality - class [quality]
def readDataset_wine():
    print("Test for dataset red wine quality")
    print()
    wine = pd.read_csv("datasetUCI/winequality-red.csv", sep=';')
    #print(wine.info())
    #print(wine['quality'].value_counts()) # - count elements class decision
    wine['quality'] = np.where(wine['quality'] > 6, 1, 0)
    X = wine.drop(columns=['quality'], axis=1)
    Y = wine['quality']

    return X, Y



def readDataset_parkinson(): #status - class
    print("Test for dataset parkinson disease")
    print()
    dataset = pd.read_csv('datasetUCI/parkinsons.data')
    dataset.drop('name', axis=1, inplace=True)
    #print(dataset.info)
    X = dataset.drop(columns=['status'], axis=1)
    Y = dataset['status']

    return X,Y

def readDataset_diabets(): # class[class]
    print("Test for dataset diabetes")

    print()
    ozone = pd.read_csv("datasetUCI/diabetes.csv")

    le = LabelEncoder()

    for i in ozone.columns:
        ozone[i] = le.fit_transform(ozone[i])

    #print(ozone.info())
    #print(ozone['class'].value_counts())

    X = ozone.drop(columns=['class'], axis=1)
    Y = ozone['class']

    return X,Y

