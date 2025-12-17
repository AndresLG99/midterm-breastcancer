from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statistics import mean

def dt_model(X_train, X_test, Y_train, Y_test, max_depths, min_samples_split_values,min_samples_leaf_values):
    base_dt = DecisionTreeClassifier(random_state=42)
    base_dt.fit(X_train, Y_train)
    Y_base = base_dt.predict(X_test)
    base_acc = accuracy_score(Y_test, Y_base)

    # Find the best value for max_depth in decision tree
    best_score = 0
    best_params = None

    for md in max_depths:
        for mss in min_samples_split_values:
            for msl in min_samples_leaf_values:
                dt = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
                random_state=42)
                cv_scores = cross_val_score(dt, X_train, Y_train, cv=5)
                avg_score = mean(cv_scores)

                print(f"max_depth={md}, min_samples_split={mss}, min_samples_leaf={msl} -> CV accuracy={avg_score:.4f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (md, mss, msl)

    best_md, best_mss, best_msl = best_params
    dt_best = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=best_md,
        min_samples_split=best_mss,
        min_samples_leaf=best_msl,
        random_state=42)
    dt_best.fit(X_train, Y_train)
    Y_best = dt_best.predict(X_test)
    test_acc = accuracy_score(Y_test, Y_best)

    print(f"\nBest params: max_depth={best_md}, min_samples_split={best_mss}, min_samples_leaf={best_msl}")
    print(f"\nAccuracy score for optimized decision tree: {test_acc:.4f}")

    return test_acc, Y_best, best_md, dt_best