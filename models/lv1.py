import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from .utils import plot_accuracies, save_reports
from .knn import ModelKNN
from .svm import ModelSVM
from .cnn import ModelCNN


# Can chay thu voi tat cac cac tham so co the
# Lam giong bai Model Selection (Slide L6, va collab cua tiet bai tap thu 4 tren lop)

# Note hom bai tap (note.txt trong folder /bt/buoi_4)

HYPER_PARAMETERS = {
    'KNN': {
        'k': list(range(1, 26)),
        'distance_fn': ["minkowski", "manhattan", "euclidean", "cosine"]
    },
    'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1.0, 2.0, 5.0, 10.0]
    },
    'CNN': {
        'optimizer': [Adam, SGD],
        'lr': [1e-2, 1e-3, 1e-4],
        'batch_size': [32, 64, 128],
        'epochs': [5, 10, 20],
    }
}



def run_knn(train_data, test_data, hyper_parameters, save_path="output/models/lv1_1k_pbm/knn"):
    """Model selection for KNN"""
    
    # Init model
    knn_model = ModelKNN()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    knn_model.prepare(X_train, X_test, y_train, y_test)
    
    # Model selection - k
    accuracies_k = []
    reports_k = []
    
    for k in hyper_parameters['k']:
        _, accuracy, report = knn_model.run(k=k)
        accuracies_k.append(accuracy)
        reports_k.append(report)
    
    filename = os.path.join(save_path, "model_selection-k")
    plot_accuracies(
        X=hyper_parameters['k'],
        y=accuracies_k,
        X_label="Number of neighbors (k)",
        y_label="Accuracy",
        title="[Model Selection] KNN - k",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='k',
        hyperparam_values=hyper_parameters['k'],
        reports=reports_k,
        save_path=filename
    )
    
    # Model selection - distance function
    accuracies_dist = []
    reports_dist = []

    for distance_fn in hyper_parameters['distance_fn']:
        _, accuracy, report = knn_model.run(distance_fn=distance_fn)
        accuracies_dist.append(accuracy)
        reports_dist.append(report)
    
    filename = os.path.join(save_path, "model_selection-distance_fn")
    plot_accuracies(
        X=hyper_parameters['distance_fn'],
        y=accuracies_dist,
        X_label="Distance functions",
        y_label="Accuracy",
        title="[Model Selection] KNN - distance function",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='distance_fn',
        hyperparam_values=hyper_parameters['distance_fn'],
        reports=reports_dist,
        save_path=filename
    )
        
    
def run_svm(train_data, test_data, hyper_parameters, save_path="output/models/lv1_1k_pbm/svm"):
    """Model selection for SVM"""

    # Init model
    svm_model = ModelSVM()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    svm_model.prepare(X_train, X_test, y_train, y_test)
    
    # Model selection - kernel
    accuracies_kernel = []
    reports_kernel = []
    
    for kernel in hyper_parameters['kernel']:
        _, accuracy, report = svm_model.run(kernel=kernel)
        accuracies_kernel.append(accuracy)
        reports_kernel.append(report)
    
    filename = os.path.join(save_path, "model_selection-kernel")
    plot_accuracies(
        X=hyper_parameters['kernel'],
        y=accuracies_kernel,
        X_label="Kernel",
        y_label="Accuracy",
        title="[Model Selection] SVM - kernel",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='kernel',
        hyperparam_values=hyper_parameters['kernel'],
        reports=reports_kernel,
        save_path=filename
    )
    
    # Model selection - regularization (C)
    accuracies_c = []
    reports_c = []

    for c in hyper_parameters['C']:
        _, accuracy, report = svm_model.run(C=c)
        accuracies_c.append(accuracy)
        reports_c.append(report)
    
    filename = os.path.join(save_path, "model_selection-c")
    plot_accuracies(
        X=hyper_parameters['C'],
        y=accuracies_c,
        X_label="Regularization (C)",
        y_label="Accuracy",
        title="[Model Selection] SVM - C",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='C',
        hyperparam_values=hyper_parameters['C'],
        reports=reports_c,
        save_path=filename
    )
    

def run_svm(train_data, test_data, hyper_parameters, save_path="output/models/lv1_1k_pbm/cnn"):
    """Model selection for CNN"""

    # Init model
    cnn_model = ModelCNN()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    cnn_model.prepare(X_train, X_test, y_train, y_test)
    
    # Model selection - kernel
    accuracies_kernel = []
    reports_kernel = []
    
    for kernel in hyper_parameters['kernel']:
        _, accuracy, report = cnn_model.run(kernel=kernel)
        accuracies_kernel.append(accuracy)
        reports_kernel.append(report)
    
    filename = os.path.join(save_path, "model_selection-kernel")
    plot_accuracies(
        X=hyper_parameters['kernel'],
        y=accuracies_kernel,
        X_label="Kernel",
        y_label="Accuracy",
        title="[Model Selection] SVM - kernel",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='kernel',
        hyperparam_values=hyper_parameters['kernel'],
        reports=reports_kernel,
        save_path=filename
    )
    
    # Model selection - regularization (C)
    accuracies_c = []
    reports_c = []

    for c in hyper_parameters['C']:
        _, accuracy, report = cnn_model.run(C=c)
        accuracies_c.append(accuracy)
        reports_c.append(report)
    
    filename = os.path.join(save_path, "model_selection-c")
    plot_accuracies(
        X=hyper_parameters['C'],
        y=accuracies_c,
        X_label="Regularization (C)",
        y_label="Accuracy",
        title="[Model Selection] SVM - C",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='C',
        hyperparam_values=hyper_parameters['C'],
        reports=reports_c,
        save_path=filename
    )
    

def run_cnn(train_data, test_data, hyper_parameters, save_path="output/models/lv1_1k_pbm/cnn"):
    """Model selection for CNN"""

    # Init model
    cnn_model = ModelCNN()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    cnn_model.prepare(X_train, X_test, y_train, y_test)

    # Model selection - optimizer
    accuracies_optimizer = []
    reports_optimizer = []
    
    optimizer_names = ['Adam', 'SGD'] # hardcode
    
    for optimizer in hyper_parameters['optimizer']:
        _, accuracy, report = cnn_model.run(optimizer=optimizer)
        accuracies_optimizer.append(accuracy)
        reports_optimizer.append(report)
        
    filename = os.path.join(save_path, "model_selection-optimizer")
    plot_accuracies(
        X=optimizer_names,
        y=accuracies_optimizer,
        X_label="Optimizer",
        y_label="Accuracy",
        title="[Model Selection] CNN - Optimizer",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='optimizer',
        hyperparam_values=optimizer_names,
        reports=reports_optimizer,
        save_path=filename
    )

    # Model selection - learning rate (lr)
    accuracies_lr = []
    reports_lr = []
    
    lr_labels = [str(lr) for lr in hyper_parameters['lr']]

    for lr in hyper_parameters['lr']:
        _, accuracy, report = cnn_model.run(lr=lr)
        accuracies_lr.append(accuracy)
        reports_lr.append(report)
        
    filename = os.path.join(save_path, "model_selection-lr")
    plot_accuracies(
        X=lr_labels,
        y=accuracies_lr,
        X_label="Learning Rate (lr)",
        y_label="Accuracy",
        title="[Model Selection] CNN - Learning Rate",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='lr',
        hyperparam_values=hyper_parameters['lr'],
        reports=reports_lr,
        save_path=filename
    )

    # Model selection - batch_size
    accuracies_batch_size = []
    reports_batch_size = []
    
    batch_size_labels = [str(bs) for bs in hyper_parameters['batch_size']]

    for batch_size in hyper_parameters['batch_size']:
        _, accuracy, report = cnn_model.run(batch_size=batch_size)
        accuracies_batch_size.append(accuracy)
        reports_batch_size.append(report)
        
    filename = os.path.join(save_path, "model_selection-batch_size")
    plot_accuracies(
        X=batch_size_labels,
        y=accuracies_batch_size,
        X_label="Batch Size",
        y_label="Accuracy",
        title="[Model Selection] CNN - Batch Size",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='batch_size',
        hyperparam_values=hyper_parameters['batch_size'],
        reports=reports_batch_size,
        save_path=filename
    )

    # Model selection - epochs
    accuracies_epochs = []
    reports_epochs = []
    
    epochs_labels = [str(ep) for ep in hyper_parameters['epochs']]

    for epochs in hyper_parameters['epochs']:
        _, accuracy, report = cnn_model.run(epochs=epochs)
        accuracies_epochs.append(accuracy)
        reports_epochs.append(report)
        
    filename = os.path.join(save_path, "model_selection-epochs")
    plot_accuracies(
        X=epochs_labels,
        y=accuracies_epochs,
        X_label="Epochs",
        y_label="Accuracy",
        title="[Model Selection] CNN - Epochs",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='epochs',
        hyperparam_values=hyper_parameters['epochs'],
        reports=reports_epochs,
        save_path=filename
    )




def models_1k_pbm(
    data_dir="output/dataset/lv1_1k_pbm/build", 
    output_dir="output/models/lv1_1k_pbm/"
):
    """
    Model selection on dataset "lv1_1k_pbm"
    """
    
    # Paths
    knn_dir = os.path.join(output_dir, 'knn')
    svm_dir = os.path.join(output_dir, 'svm')
    cnn_dir = os.path.join(output_dir, 'cnn')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(knn_dir, exist_ok=True)
    os.makedirs(svm_dir, exist_ok=True)
    os.makedirs(cnn_dir, exist_ok=True)
    
    # Dataset
    train = np.load(os.path.join(data_dir, "train.npz"))
    test = np.load(os.path.join(data_dir, "test.npz"))
    
    # Run model selection
    run_knn(train, test, HYPER_PARAMETERS["KNN"])
    run_svm(train, test, HYPER_PARAMETERS["SVM"])
    run_cnn(train, test, HYPER_PARAMETERS["CNN"])
    

