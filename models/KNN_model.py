from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

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

    return k_values, error_rates, optimal_k, min_error, result, Y_pred_ok, knn_ok