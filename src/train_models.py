from sklearn.model_selection import train_test_split

def train_model(df_encoded, target_feature_encoded, test_size):
    df_algo = df_encoded.copy()

    # Separate data into "X" and "Y"
    X = df_algo.drop(target_feature_encoded, axis=1)
    Y = df_algo[target_feature_encoded]

    # Split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    return X_train, X_test, Y_train, Y_test