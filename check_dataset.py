import pandas as pd

def readDataset_url():
    # Load data from URL using pandas read_csv method
    wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')

    # Since column headers are not correct, assign corrections to a variable for later use
    column_headers = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                      'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color Intensity',
                      'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    # Update data with corrected headers using previously defined variable
    wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None,
                       names=column_headers)


    return wine

def readDataset_plik():
    data = pd.read_csv('datasetUCI/heart.csv')

    return data
