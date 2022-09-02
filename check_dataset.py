import pandas as pd
import numpy as np

# Breast Cancer Wisconsin - class [benign_0__mal_1]
from sklearn.preprocessing import LabelEncoder


def readDataset_breastCancer():
    print("Test for dataset Breast Cancer Wisconsin")
    print()
    data = pd.read_csv("datasetUCI/cancer_classification.csv")
    #print(data.info())
    #print(data['benign_0__mal_1'].value_counts())
    return data

# heart disease - class [target]
def readDataset_heart():
    print("Test for dataset heart-disease")
    print()
    data = pd.read_csv("datasetUCI/heart.csv")
    #print(data.info())
    return data

# red wine quality - class [quality]
def readDataset_wine():
    print("Test for dataset red wine quality")
    print()
    wine = pd.read_csv("datasetUCI/winequality-red.csv")
   # print(wine['quality'].value_counts()) - count elements class decision
    wine['quality'] = np.where(wine['quality'] > 6, 1, 0)

    return wine

def readDataset_bankoteAuthencation():
    print("Test for dataset banknote")
    print()
    banknote = pd.read_csv("datasetUCI/BankNoteAuthentication.csv")
    #print(banknote.info())

    return banknote

def readDataset_onlineShopeersIntention():  #class - Revenue
    print("Test for shoppers intention")
    print()
    shoopers = pd.read_csv("datasetUCI/online_shoppers_intention.csv")
    shoopers = shoopers.drop(['VisitorType'], axis=1)
    shoopers = shoopers.drop(['Month'], axis=1)

    shoopers['Weekend'] = shoopers['Weekend'].map({False: 0, True: 1})
    shoopers['Revenue'] = shoopers['Revenue'].map({False: 0, True: 1})
    #print(shoopers.info())

    return shoopers

def readDataset_parkinson():
    print("Test for dataset parkinson")
    print()
    echo = pd.read_csv("datasetUCI/pd_speech_features.csv")
    #print(echo.info())

    return echo

def readDataset_diabets(): # class[class]
    print("Test for dataset diabets")

    print()
    ozone = pd.read_csv("datasetUCI/diabetes.csv")

    le = LabelEncoder()

    for i in ozone.columns:
        ozone[i] = le.fit_transform(ozone[i])

    print(ozone.info())

    return ozone


