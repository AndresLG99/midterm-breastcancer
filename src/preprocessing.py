import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

def drop_rows_from_col(df_original, column):
    df_rows_dropped = df_original.copy()
    df_rows_dropped = df_rows_dropped.dropna(subset=[column])
    return df_rows_dropped

def drop_unnecessary_columns(df_rows_dropped, *args):
    df_no_unnecessary = df_rows_dropped.copy()
    for column in args:
        df_no_unnecessary.drop([column], axis=1, inplace=True)
    return df_no_unnecessary

def mapping_target_feature(df_no_unnecessary, target_feature, key1, key2, value1, value2):
    df_target_feature_mapped = df_no_unnecessary.copy()
    df_target_feature_mapped[target_feature] = df_target_feature_mapped[target_feature].map({key1: value1, key2: value2})
    return df_target_feature_mapped

def find_missing_values(df_target_feature_mapped):
    missing_values_cols = []
    for column in df_target_feature_mapped.columns:
        if df_target_feature_mapped[column].isnull().sum() > 0:
            missing_values_cols.append(column)
            print(column)
            print(df_target_feature_mapped[column].isnull().sum())
    return missing_values_cols

def handle_missing_values(df_target_feature_mapped, missing_values_cols, num_cols, cat_cols):
    df_no_missing_values = df_target_feature_mapped.copy()
    for column in missing_values_cols:
        if column in num_cols:
            df_no_missing_values[column] = df_no_missing_values[column].fillna(df_no_missing_values[column].median())
        elif column in cat_cols:
            df_no_missing_values[column] = df_no_missing_values[column].fillna("Unknown")
    return df_no_missing_values

def find_numerical_features(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return num_cols

def find_categorical_features(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols

def handle_outliers(df_clipped, lower_bound, upper_bound,  k=1.5):
    q1 = df_clipped.quantile(lower_bound)
    q3 = df_clipped.quantile(upper_bound)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return df_clipped.clip(lower=lower, upper=upper)

    # How to run function
    """for c in num_cols:
        df_clipped[c] = handle_outliers(df_clipped[c])"""

def skewness_check(df_clipped, num_cols):
    skewness = df_clipped[num_cols].skew()
    print("\nSkewness (positive = right-skewed):\n")
    print(skewness)

def encode_df(df_clipped, **kwargs):
    df_encoded = df_clipped.copy()

    for encoder, values in kwargs.items():
        # 1. Ordinal encoding / Natural order
        if encoder == "ordinal":
            """if "education" in ordinal_cols:
                    order_map = sorted(df[["education", "education_num"]].drop_duplicates().sort_values("education_num")["education"].tolist())
                    ordinal_enc = OrdinalEncoder(categories=[order_map], handle_unknown="use_encoded_value", unknown_value=-1)
                    df["education"] = ordinal_enc.fit_transform(df[["education"]].fillna("missing")).ravel()"""

        # 2. Label encoding / Binary
        elif encoder == "label":
            for col in values:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).fillna("missing"))

        # 3. One-hot encoding / Nominal, no natural order
        elif encoder == "onehot":
            for col in values:
                dummies = pd.get_dummies(df_encoded[col].fillna("missing"), prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)

        # 4. Target feature encoding / Keep separate for modeling
        elif encoder == "target":
            le_target = LabelEncoder()
            df_encoded[f"{values}_encoded"] = le_target.fit_transform(df_encoded[values].astype(str).fillna("missing"))

            # Drop original target
            df = df_encoded.drop(values, axis=1)

        return df_encoded

def keep_top_features(df_encoded, top_features, target_feature):
    """
    Given a DataFrame, a list of top feature names, and the target column name,
    return a new DataFrame that contains only those columns.
    """
    cols_to_keep = top_features + [target_feature]

    # Intersect with existing columns in case of mismatch
    cols_to_keep = df_encoded.columns.intersection(cols_to_keep)
    df_reduced = df_encoded[cols_to_keep].copy()
    return df_reduced

def scaling(df_encoded, num_cols):
    df_scaled = df_encoded.copy()
    scaler = StandardScaler()
    df_scaled[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    return df_scaled

