# Run (at root path): python -m weights.save_models 

import os
import numpy as np
import torch
import joblib

from models.base import ModelBaseClass
from models.knn import ModelKNN
from models.decision_tree import ModelDecisionTree
from models.random_forest import ModelRandomForest
from models.svm import ModelSVM
from models.cnn import ModelCNN


# Variables
DATASET_NAME = "lv0_emnist"
# DATASET_NAME = "lv1_1k_pbm"

# Constants
DATA_DIR = f"output/dataset/{DATASET_NAME}/build"
SAVE_DIR = f"weights/{DATASET_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)


# Utils
def save(model, filepath):
    """Save a model instance to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


# Init models
knn_model = ModelKNN()
dt_model = ModelDecisionTree()
rf_model = ModelRandomForest()
svm_model = ModelSVM()
cnn_model = ModelCNN()

# Prepare dataset
train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']


def run_pipeline(model: ModelBaseClass, save_path, **kwargs):
    # Prepare dataset
    model.prepare(X_train, X_test, y_train, y_test)
    
    # Take the best models 
    model.run(**kwargs)

    # Save models
    save(model, filepath=save_path)


# Params are selected depends on the model selection pipeline 
# (script in `models/model_selection.py`, results in `output/models/<dataset_name>/`)

# lv0_emnist
run_pipeline(
    knn_model, f"{SAVE_DIR}/knn.joblib", 
    k=7, distance_fn="cosine", inplace=True
)
run_pipeline(
    dt_model, f"{SAVE_DIR}/decision_tree.joblib", 
    max_depth=15, inplace=True
)
run_pipeline(
    rf_model, f"{SAVE_DIR}/random_forest.joblib",
    num_trees=150, inplace=True
)
run_pipeline(
    svm_model, f"{SAVE_DIR}/svm_linear.joblib",
    C=10.0, model_type='LinearSVC', inplace=True
)
run_pipeline(
    cnn_model, f"{SAVE_DIR}/cnn.joblib",
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam, 
    epochs=10, batch_size=128, lr=1e-3, 
    inplace=True
)


# lv1_1k_pbm
# run_pipeline(
#     knn_model, f"{SAVE_DIR}/knn.joblib", 
#     k=3, distance_fn="minkowski", inplace=True
# )
# run_pipeline(
#     dt_model, f"{SAVE_DIR}/decision_tree.joblib", 
#     max_depth=10, inplace=True
# )
# run_pipeline(
#     rf_model, f"{SAVE_DIR}/random_forest.joblib",
#     num_trees=100, inplace=True
# )
# run_pipeline(
#     svm_model, f"{SAVE_DIR}/svm.joblib",
#     model_type='SVC', kernel="rbf", C=2.0, inplace=True
# )
# run_pipeline(
#     cnn_model, f"{SAVE_DIR}/cnn.joblib",
#     criterion=torch.nn.CrossEntropyLoss,
#     optimizer=torch.optim.Adam, 
#     epochs=20, batch_size=32, lr=1e-2, 
#     inplace=True
# )

