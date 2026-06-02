import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from .base import ModelBaseClass
from .utils import flatten

class ModelKNN(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        self.k = 5
        self.distance_fn = "minkowski"
        
    
    def prepare(self, X_train, X_test, y_train, y_test):
        self.X_train = flatten(X_train)
        self.X_test = flatten(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    
    def run(self, k=5, distance_fn="minkowski", inplace=False):
        model = KNeighborsClassifier(n_neighbors=k, metric=distance_fn, p=2, weights='distance')
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        if inplace:
            self.model = model
            self.k = k
            self.distance_fn = distance_fn
            self.accuracy = accuracy
            self.report = report
        
        return model, accuracy, report
        