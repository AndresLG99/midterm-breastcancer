import pandas as pd

def load_data(name_csv):
    path = f"../data/{name_csv}"
    df = pd.read_csv(path)
    return df

def show_first_rows(df, rows):
    df.head(rows)

def show_shape(df):
    rows, columns = df.shape
    print(f"Rows: {rows}\nColumns: {columns}")