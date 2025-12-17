import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import plot_tree

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

def show_confusion_matrix(Y_test, Y_pred):
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)

def show_classification_report(Y_test, Y_pred, target_name1, target_name2):
    print(classification_report(Y_test, Y_pred, target_names=[target_name1, target_name2]))

def plot_outliers(df_no_missing_values, num_cols, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    axes = axes.ravel()

    for ax, c in zip(axes, num_cols):
        sns.boxplot(y=df_no_missing_values[c], ax=ax)
        ax.set_title(c)

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df_encoded, num_cols, target_feature, k):
    """
    Plot correlation matrix for numerical columns, print top-k features
    most correlated (by absolute value) with target_feature, and return
    their names as a Python list.
    """
    # Correlation Matrix
    corr_matrix = df_encoded[num_cols].corr()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".3f",
        cbar_kws={"label": "Correlation"})
    plt.title("Correlation Matrix (Numerical Features + Target)")
    plt.tight_layout()
    plt.show()

    # Absolute correlation with target, sorted
    target_corr = (corr_matrix[target_feature].drop(target_feature).abs().sort_values(ascending=False))
    top_k = target_corr.head(k)

    print(f"Top {k} features most correlated with {target_feature}:")
    for feature, corr in top_k.items():
        print(f"{feature}: {corr:.4f}")

    # Return list of feature names
    return top_k.index.to_list()

def plot_decision_tree(X_train, target_name1, target_name2, best_depth, dt_best):
    plt.figure(figsize=(25, 15))
    plot_tree(dt_best,
              feature_names=X_train.columns,
              class_names=[target_name1, target_name2],
              filled=True,
              rounded=True,
              max_depth=best_depth,
              fontsize=10)
    plt.title(f"Decision Tree (First {best_depth} Levels)", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_elbow_graph(k_values, error_rates,optimal_k,min_error):
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, error_rates, marker="o", linestyle="dashed", color="b", markersize=4)
    plt.scatter(optimal_k, min_error, color="red", s=200, marker="*", label=f"Optimal k={optimal_k}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Error Rate")
    plt.title(f"KNN Elbow Method - Optimal k = {optimal_k} (Error: {min_error:.4f})")
    plt.xticks(np.arange(1, 100, step=2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    print(f"Optimal k: {optimal_k} (Accuracy: {1 - min_error:.4f})")