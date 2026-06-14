import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from .utils import plot_accuracies, save_reports, apply_pca
from .knn import ModelKNN
from .decision_tree import ModelDecisionTree
from .random_forest import ModelRandomForest
from .svm import ModelSVM
from .cnn import ModelCNN



HYPER_PARAMETERS = {
    'KNN': {
        'k': list(range(1, 26)),
        'distance_fn': ["minkowski", "manhattan", "euclidean", "cosine"]
    },
    'DecisionTree': {
        'max_depth': list(range(2, 21))
    },
    'RandomForest': {
        'num_trees': [5, 10, 15, 20, 30, 50, 75, 100, 150]
    },
    'SVM': {
        'SVC': {
            'model_type': 'SVC',
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1.0, 2.0, 5.0, 10.0]
        },
        'LinearSVC': {
            'model_type': 'LinearSVC',
            'C': [0.01, 0.1, 1.0, 10.0]
        }
    },
    'CNN': {
        'optimizer': [Adam, SGD],
        'lr': [1e-2, 1e-3, 1e-4],
        'batch_size': [32, 64, 128],
        'epochs': [5, 10, 20],
    }
}



DIM_REDUCTION_CONFIG = {
    "pca": {
        "func": apply_pca,
        "variance_threshold": 0.90
    }
}



def run_knn(train_data, test_data, hyper_parameters, save_path):
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



def run_dt(train_data, test_data, hyper_parameters, save_path):
    """Model selection for Decision Tree"""
    
    # Init model
    dt_model = ModelDecisionTree()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    dt_model.prepare(X_train, X_test, y_train, y_test)
    
    # Model selection - max_depth
    accuracies_depth = []
    reports_depth = []
    
    for depth in hyper_parameters['max_depth']:
        _, accuracy, report = dt_model.run(max_depth=depth)
        accuracies_depth.append(accuracy)
        reports_depth.append(report)
    
    filename = os.path.join(save_path, "model_selection-max_depth")
    plot_accuracies(
        X=hyper_parameters['max_depth'],
        y=accuracies_depth,
        X_label="Max depth",
        y_label="Accuracy",
        title="[Model Selection] Decision Tree - max depth",
        plot_type='line',
        save_path=filename
    )
    save_reports(
        hyperparam_name='max_depth',
        hyperparam_values=hyper_parameters['max_depth'],
        reports=reports_depth,
        save_path=filename
    )



def run_rf(train_data, test_data, hyper_parameters, save_path):
    """Model selection for Random Forest"""
    
    # Init model
    rf_model = ModelRandomForest()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    rf_model.prepare(X_train, X_test, y_train, y_test)
    
    # Model selection - num_trees
    accuracies_trees = []
    reports_trees = []
    
    for ntree in hyper_parameters['num_trees']:
        _, accuracy, report = rf_model.run(num_trees=ntree)
        accuracies_trees.append(accuracy)
        reports_trees.append(report)
    
    filename = os.path.join(save_path, "model_selection-num_trees")
    plot_accuracies(
        X=hyper_parameters['num_trees'],
        y=accuracies_trees,
        X_label="Number of trees",
        y_label="Accuracy",
        title="[Model Selection] Random Forest - num trees",
        plot_type='bar',
        save_path=filename
    )
    save_reports(
        hyperparam_name='num_trees',
        hyperparam_values=hyper_parameters['num_trees'],
        reports=reports_trees,
        save_path=filename
    )


    
def run_svm(train_data, test_data, hyper_parameters, save_path):
    """Model selection for SVM"""

    # Init model
    svm_model = ModelSVM()

    # Prepare dataset
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    svm_model.prepare(X_train, X_test, y_train, y_test)
    

    if hyper_parameters['model_type'] == 'SVC':
        # Model selection - kernel (only SVC)
        accuracies_kernel = []
        reports_kernel = []
        
        for kernel in hyper_parameters['kernel']:
            _, accuracy, report = svm_model.run(kernel=kernel, model_type=hyper_parameters['model_type'])
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
    
    
    if hyper_parameters['model_type'] in ['SVC', 'LinearSVC']:
        # Model selection - regularization (C) (SVC + LinearSVC)
        accuracies_c = []
        reports_c = []

        for c in hyper_parameters['C']:
            _, accuracy, report = svm_model.run(C=c, model_type=hyper_parameters['model_type'])
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



def run_cnn(train_data, test_data, hyper_parameters, save_path):
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





def model_selection_pipeline(
    data_dir="output/dataset/<dataset_name>/build", 
    output_dir="output/models/<dataset_name>/",
    dim_reduction_method=None
):
    """
    Model selection pipeline.

    Params:
        data_dir: Path to the Dataset directory (contains train.npz & test.npz)
        output_dir: Path to save outputs of this pipeline
    """
    
    # Save paths
    save_filenames = [
        'test', 'dimension_reduction', 
        'knn', 'decision_tree', 'random_forest', 
        'svm', 'linear_svm', 'cnn'
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    for save_filename in save_filenames:
        os.makedirs(os.path.join(output_dir, save_filename), exist_ok=True)
    
    # Dataset
    train = np.load(os.path.join(data_dir, "train.npz"))
    test = np.load(os.path.join(data_dir, "test.npz"))
    
    # Dimension reduction
    if dim_reduction_method:        
        dim_reduction_config = DIM_REDUCTION_CONFIG[dim_reduction_method]
        dim_reduction_func = dim_reduction_config["func"]

        train, test, final_dims = dim_reduction_func(
            train, test,
            configs=dim_reduction_config,
            save_plot_dir=os.path.join(output_dir, 'dimension_reduction')
        )
    
    
    # Run model selection

    # run_knn(train, test, HYPER_PARAMETERS["KNN"], save_path=os.path.join(output_dir, "knn"))
    # run_dt(train, test, HYPER_PARAMETERS["DecisionTree"], save_path=os.path.join(output_dir, "decision_tree"))
    # run_rf(train, test, HYPER_PARAMETERS["RandomForest"], save_path=os.path.join(output_dir, "random_forest"))
    # run_svm(train, test, HYPER_PARAMETERS["SVM"]["SVC"], save_path=os.path.join(output_dir, "svm"))
    # run_svm(train, test, HYPER_PARAMETERS["SVM"]["LinearSVC"], save_path=os.path.join(output_dir, "linear_svm"))
    # run_cnn(train, test, HYPER_PARAMETERS["CNN"], save_path=os.path.join(output_dir, "cnn"))
    
    run_knn(train, test, HYPER_PARAMETERS["KNN"], save_path=os.path.join(output_dir, "test"))
    

