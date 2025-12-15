from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

def knn_model(X_train, X_test, Y_train, Y_test, range1, range2):

    # Test different values of k
    error_rates = []
    k_values = range(range1, range2 + 1)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        error = 1 - accuracy_score(Y_test, Y_pred)
        error_rates.append(error)

    # Find optimal k
    optimal_k = k_values[np.argmin(error_rates)]
    min_error = min(error_rates)

    # Calculate test accuracy for optimal k
    knn_ok = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_ok.fit(X_train, Y_train)
    Y_pred_ok = knn_ok.predict(X_test)
    result = accuracy_score(Y_test, Y_pred_ok)
    print(f"Accuracy score for KNN with {optimal_k} neighbors : {result:.4f}")

    return k_values, error_rates, optimal_k, min_error, result, Y_pred_ok

def plot_elbow_graph(k_values, error_rates,optimal_k,min_error):
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, error_rates, marker="o", linestyle="dashed", color="b", markersize=4)
    plt.scatter(optimal_k, min_error, color="red", s=200, marker="*", label=f"Optimal k={optimal_k}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Error Rate")
    plt.title(f"KNN Elbow Method - Optimal k = {optimal_k} (Error: {min_error:.4f})")
    plt.xticks(np.arange(1, 101, step=5))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    print(f"Optimal k: {optimal_k} (Accuracy: {1 - min_error:.4f})")

def show_confusion_matrix(Y_test, Y_pred):
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)

def show_classification_report(Y_test, Y_pred, target_name1, target_name2):
    print(classification_report(Y_test, Y_pred, target_names=[target_name1, target_name2]))