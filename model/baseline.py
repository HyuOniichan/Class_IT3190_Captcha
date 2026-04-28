import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def flatten(X):
    """Flatten (N, 28, 28) → (N, 784) and normalize to [0, 1]."""
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0


def run_knn(X_train, y_train, X_test, y_test, k=5):
    X_tr = flatten(X_train)
    X_te = flatten(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"KNN (k={k})  —  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(report)

    return model, acc, report


def run_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=1.0):
    X_tr = flatten(X_train)
    X_te = flatten(X_test)

    model = SVC(kernel=kernel, C=C, gamma="scale")
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"SVM (kernel={kernel}, C={C})  —  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(report)

    return model, acc, report


def run_baselines(X_train, y_train, X_test, y_test):
    """Run both KNN and SVM and return a comparison dict."""
    print("\n[Baseline] Running KNN …")
    knn_model, knn_acc, _ = run_knn(X_train, y_train, X_test, y_test)

    print("\n[Baseline] Running SVM …")
    svm_model, svm_acc, _ = run_svm(X_train, y_train, X_test, y_test)

    results = {
        "knn": {"model": knn_model, "accuracy": knn_acc},
        "svm": {"model": svm_model, "accuracy": svm_acc},
    }

    print(f"\n{'='*50}")
    print(f"Baseline Summary")
    print(f"  KNN accuracy : {knn_acc:.4f}")
    print(f"  SVM accuracy : {svm_acc:.4f}")
    print(f"{'='*50}")

    return results


if __name__ == "__main__":
    data_dir = "dataset/ready/1k_pbm"
    train = np.load(f"{data_dir}/train.npz")
    test = np.load(f"{data_dir}/test.npz")
    run_baselines(train["X"], train["y"], test["X"], test["y"])
