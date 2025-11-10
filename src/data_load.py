import pandas as pd

def load_data(path="../data/data.csv"):
    df = pd.read_csv(path)
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
    return df