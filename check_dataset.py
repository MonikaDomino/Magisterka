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
    #print(data['benign_0__mal_1'].value_counts())
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

# bankonte - class [class]
def readDataset_bankoteAuthencation():
    print("Test for dataset banknote")
    print()
    banknote = pd.read_csv("datasetUCI/BankNoteAuthentication.csv")
    #print(banknote.info())
    X = banknote.drop(columns=['class'], axis=1)
    Y = banknote['class']

    return X,Y

def readDataset_onlineShopeersIntention():  #class - Revenue
    print("Test for shoppers intention")
    print()
    shoopers = pd.read_csv("datasetUCI/online_shoppers_intention.csv")
    shoopers = shoopers.drop(['VisitorType'], axis=1)
    shoopers = shoopers.drop(['Month'], axis=1)

    shoopers['Weekend'] = shoopers['Weekend'].map({False: 0, True: 1})
    shoopers['Revenue'] = shoopers['Revenue'].map({False: 0, True: 1})
    #print(shoopers.info())

    X = shoopers.drop(columns=['Revenue'], axis=1)
    Y = shoopers['Revenue']

    return X,Y

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
    X = ozone.drop(columns=['class'], axis=1)
    Y = ozone['class']

    return X,Y

# audit risk - Risk (Class)
def readDataset_auditRisk():
    print("Test for dataset audit risk")
    print()
    risk = pd.read_csv('datasetUCI/audit_risk.csv')

    risk['Money_Value'] = risk['Money_Value'].fillna(risk['Money_Value'].mean())

    risk = risk.drop('LOCATION_ID', axis = 1)

    X = risk.drop(columns=['Risk'], axis=1)
    Y = risk['Risk']

    return X, Y


def readDataset_MHR (): # RiskLevel
    print("Test for dataset MH risk")
    print()

    mh = pd.read_csv('datasetUCI/Maternal Health Risk Data Set.csv')

    #print(mh.info())
    X = mh.drop(columns=['RiskLevel'], axis=1)
    Y = mh['RiskLevel']

    return X, Y

def readDataset_tic_tac_toe ():  # class
    print("Test for dataset BT")
    print()

    mh = pd.read_csv('datasetUCI/tic-tac-toe.data')
    mh.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square',
                  'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'Class']
    mh.replace('negative', 0, inplace=True)
    mh.replace('positive', 1, inplace=True)

    #print(mh.info())

    X = pd.get_dummies(mh[['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square',
                  'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']], drop_first=True)



   # X = mh.drop(columns=['Class'], axis=1)
    Y = mh['Class']

    return X,Y