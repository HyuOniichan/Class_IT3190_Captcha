from abc import ABC, abstractmethod

class ModelBaseClass(ABC):
    def __init__(self):
        super().__init__()
        
        self.model = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.accuracy = 0.0
        self.report = ''
    
    @abstractmethod
    def prepare(self, X_train, X_test, y_train, y_test) -> None:
        """Prepare dataset (train, test)"""
        pass

    @abstractmethod
    def run(self, model=None, inplace=False, **kwargs):
        """
        Run model.
        Params:
            model (class): Use model from parameters, or self.model if None
            inplace (bool): Assign current running model to self.model
            **kwargs: Hyperparameters for specific models
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict a sample.
        Params:
            X: Input
        """
        
    
