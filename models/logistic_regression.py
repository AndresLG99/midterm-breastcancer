from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train, y_train)
    return model

def evaluate_logistic_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report