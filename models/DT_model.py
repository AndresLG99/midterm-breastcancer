from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def dt_model(X_train, X_test, Y_train, Y_test, range1, range2):
    dt_clr = DecisionTreeClassifier()
    dt_clr.fit(X_train, Y_train)
    Y_dt_pred = dt_clr.predict(X_test)
    dt_accuracy = accuracy_score(Y_test, Y_dt_pred)

    # Find the best value for max_depth in decision tree
    scores = []
    depths = range(range1, range2)
    for d in depths:
        score = cross_val_score(DecisionTreeClassifier(criterion="entropy", max_depth=d), X_train, Y_train, cv=5)
        avg_score = mean(score)
        scores.append(avg_score)

    best_depth = depths[np.argmax(scores)]

    dt_clr_opt = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
    dt_clr_opt.fit(X_train, Y_train)

    Y_dt_opt_pred = dt_clr_opt.predict(X_test)
    result = accuracy_score(Y_test, Y_dt_opt_pred)
    print(f"Accuracy score for the optimized decision tree classifier: {result:.4f}")

    return result, dt_clr_opt, Y_dt_opt_pred

def plot_decision_tree(X_train, target_name1, target_name2, best_depth, dt_clr_opt):
    plt.figure(figsize=(25, 15))
    plot_tree(dt_clr_opt,
              feature_names=X_train.columns,
              class_names=[target_name1, target_name2],
              filled=True,
              rounded=True,
              max_depth=best_depth,
              fontsize=10)
    plt.title(f"Decision Tree (First {best_depth} Levels)", fontsize=16)
    plt.tight_layout()
    plt.show()

def show_confusion_matrix(Y_test, Y_pred):
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)

def show_classification_report(Y_test, Y_pred, target_name1, target_name2):
    print(classification_report(Y_test, Y_pred, target_names=[target_name1, target_name2]))