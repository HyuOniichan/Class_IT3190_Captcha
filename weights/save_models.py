# Run (at root path): python -m weights.save_models 

import os
import numpy as np
import joblib

from models.base import ModelBaseClass
from models.knn import ModelKNN
from models.decision_tree import ModelDecisionTree
from models.random_forest import ModelRandomForest
from models.svm import ModelSVM
from models.cnn import ModelCNN

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
DATA_DIR = "output/dataset/lv1_1k_pbm/build"
train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']


def run_pipeline(model: ModelBaseClass, save_path, **kwargs):
    # Prepare dataset
    model.prepare(X_train, X_test, y_train, y_test)
    
    # Take the best models 
    # (script in `models/lv1.py`, results in `output/models/lv1_1k_pbm/`)
    model.run(**kwargs)

    # Save models
    save(model, filepath=save_path)


run_pipeline(
    knn_model, "weights/lv1_knn_model.joblib", 
    k=10, distance_fn="minkowski", inplace=True
)
run_pipeline(
    dt_model, "weights/lv1_dt_model.joblib", 
    max_depth=10, inplace=True
)
run_pipeline(
    rf_model, "weights/lv1_rf_model.joblib",
    num_trees=100, inplace=True
)
run_pipeline(
    svm_model, "weights/lv1_svm_model.joblib",
    kernel="rbf", C=2.0, inplace=True
)
run_pipeline(
    cnn_model, "weights/lv1_cnn_model.joblib",
    inplace=True
)

