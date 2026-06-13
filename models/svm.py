import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from .base import ModelBaseClass
from .utils import flatten

class ModelSVM():
    def __init__(self):
        super().__init__()
        
        self.kernel = 'rbf'
        self.C = 1.0
        
    
    def prepare(self, X_train, X_test, y_train, y_test):
        self.X_train = flatten(X_train)
        self.X_test = flatten(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    
    def run(self, kernel='rbf', C=1.0, model_type='SVC', inplace=False):
        if model_type == 'SVC':
            model = SVC(kernel=kernel, C=C, probability=True)
        elif model_type == 'LinearSVC':
            # n_samples = 112800 > n_features = 784 -> dual = False (faster)
            model = LinearSVC(C=C, dual=False, max_iter=5000)
        
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        if inplace:
            self.model = model
            self.kernel = kernel
            self.C = C
            self.accuracy = accuracy
            self.report = report
        
        return model, accuracy, report
    
    def predict(self, X):
        input_tensor = flatten(X)
        y_pred = self.model.predict(input_tensor)
        
        # LinearSVC doesn't support predict probabilities
        # -> Pretend probs as an onehot array
        if isinstance(self.model, LinearSVC):
            n_classes = len(self.model.classes_)

            probs = np.zeros((len(y_pred), n_classes), dtype=np.float32)

            class_to_idx = {
                cls: idx
                for idx, cls in enumerate(self.model.classes_)
            }

            probs[
                np.arange(len(y_pred)),
                [class_to_idx[y] for y in y_pred]
            ] = 1.0

        else:
            probs = self.model.predict_proba(input_tensor)

        
        return y_pred, probs
    