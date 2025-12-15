from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def svm_model(X_train, X_test, Y_train, Y_test):
    kernels = ["rbf", "linear", "poly"]
    c = [1,100,1_000]
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_svc_pred = svc.predict(X_test)
    results = []
    max_result = 0

    for kernel in kernels:
        for i in c:
            svc_opt_linear = SVC(kernel=kernel, C=i)
            svc_opt_linear.fit(X_train, Y_train)
            Y_pred = svc_opt_linear.predict(X_test)
            result = accuracy_score(Y_test, Y_pred)
            results.append(result)
            print(f"Accuracy score for C={c} and {kernel} kernel: {result:.4f}")
            if result > max_result:
                max_result = result
                max_kernel = kernel
                max_c = i

    svc_opt = SVC(kernel=max_kernel, C=max_c, probability=True)
    svc_opt.fit(X_train, Y_train)
    Y_pred = svc_opt.predict(X_test)

    return max_result, Y_pred

def show_confusion_matrix(Y_test, Y_pred):
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)

def show_classification_report(Y_test, Y_pred, target_name1, target_name2):
    print(classification_report(Y_test, Y_pred, target_names=[target_name1, target_name2]))