import pandas as pd

def load_data(name_csv):
    path = f"../data/{name_csv}"
    df = pd.read_csv(path)
    return df

def show_shape(df):
    rows, columns = df.shape
    print(f"Rows: {rows}\nColumns: {columns}")

"""
# Create CSV file from the UCI ML Repository data

from ucimlrepo import fetch_ucirepo
# fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

X = breast_cancer_wisconsin_original.data.features   # DataFrame
Y = breast_cancer_wisconsin_original.data.targets    # DataFrame or Series

# combine features + target into one DataFrame
df = pd.concat([X, Y], axis=1)      # columns align by index

# save to CSV (no index column in file)
df.to_csv("data_original.csv", index=False)
"""