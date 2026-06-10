# Run (at root path): python -m weights.save_models 

import os
import numpy as np
import joblib

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

knn_model.prepare(X_train, X_test, y_train, y_test)
dt_model.prepare(X_train, X_test, y_train, y_test)
rf_model.prepare(X_train, X_test, y_train, y_test)
svm_model.prepare(X_train, X_test, y_train, y_test)
cnn_model.prepare(X_train, X_test, y_train, y_test)


# Take the best models 
# (script in `models/lv1.py`, results in `output/models/lv1_1k_pbm/`)
knn_model.run(k=10, distance_fn="minkowski", inplace=True)
dt_model.run(max_depth=10, inplace=True)
rf_model.run(num_trees=100, inplace=True)
svm_model.run(kernel="rbf", C=2.0, inplace=True)
cnn_model.run(inplace=True)


# Save models
save(model=knn_model, filepath="weights/lv1_knn_model.joblib")
save(model=dt_model, filepath="weights/lv1_dt_model.joblib")
save(model=rf_model, filepath="weights/lv1_rf_model.joblib")
save(model=svm_model, filepath="weights/lv1_svm_model.joblib")
save(model=cnn_model, filepath="weights/lv1_cnn_model.joblib")

