import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib



# ------------------------------
# KNN + SVM
# ------------------------------
def flatten(X):
    """Flatten (N, 28, 28) → (N, 784) and normalize to [0, 1]."""
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0





# ------------------------------
# Simple CNN
# ------------------------------
def _prepare_tensors(X, y=None, label_map=None):
    """Convert numpy arrays to torch tensors with proper shape/dtype."""
    
    X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0   # (N, 1, 28, 28)

    if y is None:
        return X_t, None

    if label_map is not None:
        y = np.vectorize(label_map.get)(y)
        
    y_t = torch.from_numpy(y).long()

    return X_t, y_t

def _make_loader(X_t, y_t, batch_size, shuffle):
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)





# ------------------------------
# Helpers
# ------------------------------
def plot_accuracies(X, y, X_label, y_label, title, plot_type='line', save_path=""):
    """
    Plot and save accuracies. Utils for Model Selection section.
    """
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Draw lines
    if plot_type == 'bar':
        ax.bar(X, y, label='Accuracy', color='skyblue', edgecolor='black', alpha=0.8)
    else:
        ax.plot(X, y, marker='o', linestyle='-', label='Accuracy')
    
    for val, acc_val in zip(X, y):
        offset = 0.005 if plot_type == 'line' else (max(y) * 0.01)
        ax.text(
            val, acc_val + offset, f"{acc_val:.4f}",
            ha='center', va='bottom',
            fontsize=9,
            fontweight='bold',
            color='darkblue'
        )
    
    ax.set_xlabel(X_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.set_xticks(X)
    ax.set_ylim(min(y) - 0.02, max(y) + 0.03)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Model Selection] Plot {title} saved.")
    
    plt.show()


def save_reports(hyperparam_name, hyperparam_values, reports, save_path):
    """
    Save classification reports. Utils for Model Selection section.
    """
    if not save_path.endswith('.csv'):
        save_path = f"{save_path}.csv"
    
    if isinstance(reports[0], dict):
        rows = []
        for val, report in zip(hyperparam_values, reports):
            for class_name, metrics in report.items():
                if isinstance(metrics, dict): 
                    rows.append({
                        hyperparam_name: val,
                        'class': class_name,
                        'precision': metrics.get('precision'),
                        'recall': metrics.get('recall'),
                        'f1-score': metrics.get('f1-score'),
                        'support': metrics.get('support')
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"[Model Selection] Report {hyperparam_name} saved")
    else:
        # String type
        with open(save_path, 'w', encoding='utf-8') as f:
            for val, report in zip(hyperparam_values, reports):
                f.write(f"{'='*30}\n")
                f.write(f" Report for {hyperparam_name.upper()} = {val}\n")
                f.write(f"{'='*30}\n")
                f.write(str(report))
                f.write("\n\n")
        print(f"[Model Selection] Report {hyperparam_name} saved")
        




# ------------------------------
# Dimension Reduction
# ------------------------------
def apply_pca(train_data, test_data, configs, save_plot_dir=None, save_transformer_path=None):
    """
    Hàm đóng gói xử lý PCA.
    Configs yêu cầu: {'variance_threshold': float} (ví dụ: 0.90)
    """
    variance_threshold = configs.get('variance_threshold', 0.90)
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Đảm bảo dữ liệu phẳng (Flatten)
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"\n[Dim Reduction] (PCA) Đang tính toán PCA cho dữ liệu gốc {X_train.shape}...")
    
    # 1. Khởi tạo PCA tổng thể để phân tích phương sai tích lũy
    pca_full = PCA()
    pca_full.fit(X_train)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Tìm số chiều tối ưu dựa trên ngưỡng threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"[Dim Reduction] (PCA) Giữ lại {variance_threshold*100}% thông tin -> Cần: {n_components} chiều.")
    
    # 2. Vẽ và lưu đồ thị phân tích cho báo cáo
    if save_plot_dir:
        os.makedirs(save_plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, linewidth=2, color='b')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100}% Variance')
        plt.axvline(x=n_components, color='g', linestyle='--', label=f'n_components = {n_components}')
        plt.xlabel('Số lượng thành phần chính (Components)')
        plt.ylabel('Phương sai tích lũy (Cumulative Explained Variance)')
        plt.title('Phân tích phương sai tích lũy bằng PCA')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(save_plot_dir, "pca_variance_analysis.png"), dpi=300)
        plt.close()
        print(f"[Dim Reduction] (PCA) Đã lưu biểu đồ phân tích tại: {save_plot_dir}/pca_variance_analysis.png")
    

    # 3. Tiến hành giảm chiều thực tế với n_components tối ưu
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Save the PCA transformer
    if save_transformer_path:
        joblib.dump(pca, save_transformer_path)
        print(f"[Dim Reduction] (PCA) Đã lưu model PCA: {save_transformer_path}")
    
    # Đóng gói kết quả tương thích với cấu trúc pipeline hiện tại
    train_reduced = {'X': X_train_pca, 'y': y_train}
    test_reduced = {'X': X_test_pca, 'y': y_test}
    
    return train_reduced, test_reduced, n_components



