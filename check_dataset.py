import pandas as pd

def readDataset(path):
    # read data from file
    data = pd.read_csv(path)
    # print(data.info())
    # print(data.status.value_counts())

    return data
