from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm_model(X_train, X_test, Y_train, Y_test, kernels, cs, gammas):
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_svc_pred = svc.predict(X_test)

    results = []
    max_result = 0
    max_kernel = None
    max_c = None
    max_gamma = None

    for kernel in kernels:
        for c in cs:
            for g in gammas:
                svc_opt = SVC(kernel=kernel, C=c, gamma=g)
                svc_opt.fit(X_train, Y_train)
                Y_pred = svc_opt.predict(X_test)
                result = accuracy_score(Y_test, Y_pred)
                results.append(result)
                print(f"Accuracy score for C={c}, kernel={kernel} and gamma={g}: {result:.4f}")
                if result > max_result:
                    max_result = result
                    max_kernel = kernel
                    max_c = c
                    max_gamma = g

    svc_opt = SVC(kernel=max_kernel, C=max_c, gamma=max_gamma, probability=True)
    svc_opt.fit(X_train, Y_train)
    Y_pred = svc_opt.predict(X_test)

    print(f"\nAccuracy score for best result overall:\nC={max_c}, kernel={max_kernel} and gamma={max_gamma}: {max_result:.4f}")

    return max_result, Y_pred, svc_opt