from src.data_load import load_data, convert_to_csv
from models.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from models.decision_tree import train_decision_tree, evaluate_decision_tree
from sklearn.model_selection import train_test_split

# Only use if csv is not created
# convert_to_csv()

# Load clean data
df = load_data()

# Split features/labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Logistic Regression ----------------
log_model = train_logistic_regression(X_train, y_train)
log_acc, log_report = evaluate_logistic_regression(log_model, X_test, y_test)
print("\n===== Logistic Regression =====")
print("Accuracy:", log_acc)
print(log_report)

# ---------------- Decision Tree ----------------
dt_model = train_decision_tree(X_train, y_train)
dt_acc, dt_report = evaluate_decision_tree(dt_model, X_test, y_test)
print("\n===== Decision Tree =====")
print("Accuracy:", dt_acc)
print(dt_report)