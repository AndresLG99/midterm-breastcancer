import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_load import load_data
from preprocessing import preprocess_data

def train_and_evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        })
        joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv("../results/metrics_summary.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    train_and_evaluate()