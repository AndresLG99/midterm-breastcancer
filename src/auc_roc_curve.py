from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(X_test, Y_test, optimal_k, lr, knn_ok, dt_clr_opt, svc_opt, nn):
    Y_lr_probs = lr.predict_proba(X_test)[:, 1]
    Y_knn_ok_probs = knn_ok.predict_proba(X_test)[:, 1]
    Y_dt_probs = dt_clr_opt.predict_proba(X_test)[:, 1]
    Y_svm_probs = svc_opt.predict_proba(X_test)[:, 1]
    Y_nn_probs = nn.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC for all the models
    fpr_lr, tpr_lr, _ = roc_curve(Y_test, Y_lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)

    fpr_knn, tpr_knn, _ = roc_curve(Y_test, Y_knn_ok_probs)
    auc_knn = auc(fpr_knn, tpr_knn)

    fpr_dt, tpr_dt, _ = roc_curve(Y_test, Y_dt_probs)
    auc_dt = auc(fpr_dt, tpr_dt)

    fpr_svm, tpr_svm, _ = roc_curve(Y_test, Y_svm_probs)
    auc_svm = auc(fpr_svm, tpr_svm)

    fpr_nn, tpr_nn, _ = roc_curve(Y_test, Y_nn_probs)
    auc_nn = auc(fpr_nn, tpr_nn)

    # Plot both ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_lr, tpr_lr, color="blue", label=f"Logistic Regression (AUC = {auc_lr:.3f})")
    plt.plot(fpr_knn, tpr_knn, color="red", linestyle="dashed", label=f"KNN (k={optimal_k}) (AUC = {auc_knn:.3f})")
    plt.plot(fpr_dt, tpr_dt, color="green", linestyle="dashdot", label=f"Decision Tree (AUC = {auc_dt:.3f})")
    plt.plot(fpr_svm, tpr_svm, color="orange", linestyle="dotted", label=f"SVM (AUC = {auc_svm:.3f})")
    plt.plot(fpr_nn, tpr_nn, color="purple", linestyle="-", linewidth=2.5, label=f"Neural Network (AUC = {auc_nn:.3f})")

    # Random guess line
    plt.plot([0, 1], [0, 1], color="black", linestyle=":", label="Random Guess")

    # Labels and Title
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve Comparison: All 5 Classifiers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print AUC scores for easy comparison
    print("ROC-AUC Scores:")
    print(f"Logistic Regression: {auc_lr:.3f}")
    print(f"KNN: {auc_knn:.3f}")
    print(f"Decision Tree: {auc_dt:.3f}")
    print(f"SVM: {auc_svm:.3f}")
    print(f"Neural Network: {auc_nn:.3f}")