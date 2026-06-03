import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from .base import ModelBaseClass
from .utils import flatten

class ModelRandomForest(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        self.random_state = 42
        self.num_trees = 30
        
    
    def prepare(self, X_train, X_test, y_train, y_test):
        self.X_train = flatten(X_train)
        self.X_test = flatten(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    
    def run(self, num_trees=30, inplace=False):
        model = RandomForestClassifier(random_state=self.random_state, n_estimators=num_trees)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        if inplace:
            self.model = model
            self.num_trees = num_trees
            self.accuracy = accuracy
            self.report = report
        
        return model, accuracy, report
        