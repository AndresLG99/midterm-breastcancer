import pandas as pd
from sklearn.utils import resample

def random_over_sampling(df,target_feature):
    # Split majority / minority
    counts = df[target_feature].value_counts()
    maj_class = counts.idxmax()
    min_class = counts.idxmin()

    df_majority = df[df[target_feature] == maj_class]
    df_minority = df[df[target_feature] == min_class]

    # oversample minority to match majority
    df_minority_over = resample(
        df_minority,
        replace=True,                   # sample with replacement
        n_samples=len(df_majority),     # match majority size
        random_state=42)

    df_balanced = pd.concat([df_majority, df_minority_over],
                            axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced