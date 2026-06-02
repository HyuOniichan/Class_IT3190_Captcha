import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

from preprocess.utils import INDEX_TO_CHAR

# ------------------------------
# KNN + SVM
# ------------------------------
def flatten(X):
    """Flatten (N, 28, 28) → (N, 784) and normalize to [0, 1]."""
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0





# ------------------------------
# Simple CNN
# ------------------------------

def _prepare_tensors(X, y):
    """Convert numpy arrays to torch tensors with proper shape/dtype."""
    X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0   # (N, 1, 28, 28)
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
        


