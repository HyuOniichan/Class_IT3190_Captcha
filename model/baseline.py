import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from model.metrics import compute_image_accuracy


def flatten(X):
    """Flatten (N, 28, 28) → (N, 784) and normalize to [0, 1]."""
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0


def run_knn(X_train, y_train, X_test, y_test, k=5, chars_per_image=4):
    X_tr = flatten(X_train)
    X_te = flatten(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    char_acc = accuracy_score(y_test, y_pred)
    img_acc, n_images = compute_image_accuracy(y_pred, y_test, chars_per_image)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"KNN (k={k})")
    print(f"  Per-character accuracy : {char_acc:.4f}")
    print(f"  Per-image accuracy     : {img_acc:.4f}  "
          f"({int(img_acc * n_images)}/{n_images} images)")
    print(f"{'='*60}")
    print(report)

    return model, char_acc, img_acc, report


def run_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=1.0, chars_per_image=4):
    X_tr = flatten(X_train)
    X_te = flatten(X_test)

    model = SVC(kernel=kernel, C=C, gamma="scale")
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    char_acc = accuracy_score(y_test, y_pred)
    img_acc, n_images = compute_image_accuracy(y_pred, y_test, chars_per_image)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"SVM (kernel={kernel}, C={C})")
    print(f"  Per-character accuracy : {char_acc:.4f}")
    print(f"  Per-image accuracy     : {img_acc:.4f}  "
          f"({int(img_acc * n_images)}/{n_images} images)")
    print(f"{'='*60}")
    print(report)

    return model, char_acc, img_acc, report


def run_baselines(X_train, y_train, X_test, y_test, chars_per_image=4):
    """Run both KNN and SVM and return a comparison dict."""
    print("\n[Baseline] Running KNN …")
    knn_model, knn_char, knn_img, _ = run_knn(
        X_train, y_train, X_test, y_test, chars_per_image=chars_per_image
    )

    print("\n[Baseline] Running SVM …")
    svm_model, svm_char, svm_img, _ = run_svm(
        X_train, y_train, X_test, y_test, chars_per_image=chars_per_image
    )

    results = {
        "knn": {"model": knn_model, "char_acc": knn_char, "img_acc": knn_img},
        "svm": {"model": svm_model, "char_acc": svm_char, "img_acc": svm_img},
    }

    print(f"\n{'='*60}")
    print(f"Baseline Summary")
    print(f"  KNN  char_acc={knn_char:.4f}  img_acc={knn_img:.4f}")
    print(f"  SVM  char_acc={svm_char:.4f}  img_acc={svm_img:.4f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    data_dir = "dataset/ready/1k_pbm"
    train = np.load(f"{data_dir}/train.npz")
    test = np.load(f"{data_dir}/test.npz")
    run_baselines(train["X"], train["y"], test["X"], test["y"])
