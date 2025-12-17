from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def nn_model(X_train, X_test, Y_train, Y_test):
    nn = MLPClassifier(
        random_state=42,
        max_iter=1000,              # More iterations than LR due to gradient descent
        hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
        alpha=0.001,                # L2 regularization (like LR's C parameter)
        learning_rate="adaptive"    # Adaptive learning rate for better convergence
    )
    nn.fit(X_train, Y_train)
    Y_pred_nn = nn.predict(X_test)
    Y_pred_proba_nn = nn.predict_proba(X_test)[:, 1]
    result = roc_auc_score(Y_test, Y_pred_proba_nn)
    print(f"Neural Network ROC-AUC: {result:.4f}")

    return result, Y_pred_proba_nn, Y_pred_nn, nn