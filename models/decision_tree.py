import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from .base import ModelBaseClass
from .utils import flatten

class ModelDecisionTree(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        self.random_state = 42
        self.max_depth = 10
        
    
    def prepare(self, X_train, X_test, y_train, y_test):
        self.X_train = flatten(X_train)
        self.X_test = flatten(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    
    def run(self, max_depth=10, inplace=False):
        model = DecisionTreeClassifier(random_state=self.random_state, max_depth=max_depth)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        if inplace:
            self.model = model
            self.max_depth = max_depth
            self.accuracy = accuracy
            self.report = report
        
        return model, accuracy, report
        