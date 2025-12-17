from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def lr_model(X_train, X_test, Y_train, Y_test, iterations):
    lr = LogisticRegression(random_state=42, max_iter=iterations)
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    Y_probs = lr.predict_proba(X_test)
    Y_probs = Y_probs[:, 1]
    result = roc_auc_score(Y_test, Y_probs)
    print(f"Logistic Regression ROC-AUC: {result:.4f}")
    return result, Y_probs, Y_pred, lr