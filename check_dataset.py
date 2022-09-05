import pandas as pd
import numpy as np
import openpyxl

from sklearn.preprocessing import LabelEncoder

# Breast Cancer Wisconsin - class [benign_0__mal_1]
def readDataset_breastCancer():
    print("Test for dataset Breast Cancer Wisconsin - Diagnosic")
    print()
    data = pd.read_csv("datasetUCI/cancer_classification.csv")
    print(data.info())
    #print(data['benign_0__mal_1'].value_counts())
    return data

# heart disease - class [target]
def readDataset_heart():
    print("Test for dataset heart-disease")
    print()
    data = pd.read_csv("datasetUCI/heart.csv")
    # print(data.info())
    return data

# red wine quality - class [quality]
def readDataset_wine():
    print("Test for dataset red wine quality")
    print()
    wine = pd.read_csv("datasetUCI/winequality-red.csv", sep=';')
    #print(wine.info())
    #print(wine['quality'].value_counts()) # - count elements class decision
    wine['quality'] = np.where(wine['quality'] > 6, 1, 0)

    return wine

# bankonte - class [class]
def readDataset_bankoteAuthencation():
    print("Test for dataset banknote")
    print()
    banknote = pd.read_csv("datasetUCI/BankNoteAuthentication.csv")
    print(banknote.info())

    return banknote

def readDataset_onlineShopeersIntention():  #class - Revenue
    print("Test for shoppers intention")
    print()
    shoopers = pd.read_csv("datasetUCI/online_shoppers_intention.csv")
    shoopers = shoopers.drop(['VisitorType'], axis=1)
    shoopers = shoopers.drop(['Month'], axis=1)

    shoopers['Weekend'] = shoopers['Weekend'].map({False: 0, True: 1})
    shoopers['Revenue'] = shoopers['Revenue'].map({False: 0, True: 1})
    print(shoopers.info())

    return shoopers

def readDataset_parkinson(): #status - class
    print("Test for dataset parkinson disease")
    print()
    dataset = pd.read_csv('datasetUCI/parkinsons.data')
    dataset.drop('name', axis=1, inplace=True)
    #print(dataset.info)

    return dataset

def readDataset_diabets(): # class[class]
    print("Test for dataset diabetes")

    print()
    ozone = pd.read_csv("datasetUCI/diabetes.csv")

    le = LabelEncoder()

    for i in ozone.columns:
        ozone[i] = le.fit_transform(ozone[i])

    #print(ozone.info())

    return ozone

# audit risk - Risk (Class)
def readDataset_auditRisk():
    print("Test for dataset audit risk")
    print()
    risk = pd.read_csv('datasetUCI/audit_risk.csv')

    risk['Money_Value'] = risk['Money_Value'].fillna(risk['Money_Value'].mean())

    location_dummies = pd.get_dummies(risk['LOCATION_ID'], prefix='location')
    risk = pd.concat([risk, location_dummies], axis=1)
    risk = risk.drop('LOCATION_ID', axis = 1)

    return risk


def readDataset_MHR (): # RiskLevel
    print("Test for dataset MH risk")
    print()

    mh = pd.read_csv('datasetUCI/Maternal Health Risk Data Set.csv')

    #print(mh.info())

    return mh

def readDataset_BT ():  # class
    print("Test for dataset BT")
    print()

    mh = pd.read_excel('datasetUCI/BreastTissue.xlsx')

    #print(mh.info())

    return mh