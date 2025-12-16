import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

def plot_parallel(df, target_feature):
    # Parallel Coordinate Plot
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.drop(target_feature, axis=1))
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    df_scaled[target_feature] = df[target_feature]

    # Convert class to label names for color legend
    df_scaled[target_feature] = df_scaled[target_feature].map({0: "Benign", 1: "Malignant"})

    # Create the parallel coordinates plot
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df_scaled, target_feature, color=['#1f77b4', '#d62728'], alpha=0.4)
    plt.title("Parallel Coordinate Plot - Breast Cancer Wisconsin (Original)")
    plt.xlabel("Features")
    plt.ylabel("Scaled Feature Values")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()

def plot_distribution(df, target_feature):
    sns.set(style="whitegrid")

    # all feature columns except the target
    features = [col for col in df.columns if col != target_feature]
    n_features = len(features)

    # choose a roughly square grid
    n_cols = 5  # or any fixed max columns you like
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, feature in enumerate(features, start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(
            data=df,
            x=feature,
            hue=target_feature,
            fill=True,
            common_norm=False,
            palette={0: "#1f77b4", 1: "#d62728"},
        )
        plt.title(feature)
        plt.xlabel("")
        plt.ylabel("Density")

    plt.suptitle(
        "Distribution of Each Feature for Benign and Malignant Samples",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()