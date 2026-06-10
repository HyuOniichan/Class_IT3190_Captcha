import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn

from .base import ModelBaseClass
from .utils import _prepare_tensors, _make_loader
from preprocess.utils import INDEX_TO_CHAR

class ModelCNN(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        self.save_path = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_classes = 36
        self.epochs = 20
        self.batch_size = 64
        self.lr = 1e-3
        
        self.criterion = nn.CrossEntropyLoss
        self.optimizer = torch.optim.Adam
        
    
    def prepare(self, X_train, X_test, y_train, y_test):
        self.X_train, self.y_train = _prepare_tensors(X_train, y_train) 
        self.X_test, self.y_test = _prepare_tensors(X_test, y_test) 
    
    
    def run(
        self, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, 
        num_classes=36, epochs=20, batch_size=64, lr=1e-3, 
        device='cpu', inplace=False
    ):
        model = SimpleCNN(num_classes).to(device)
        model.train()
        
        train_loader = _make_loader(self.X_train, self.y_train, batch_size, shuffle=True)
        test_loader = _make_loader(self.X_test, self.y_test, batch_size, shuffle=False)
        
        criterion = criterion()
        optimizer = optimizer(model.parameters(), lr=lr)
        
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += X_batch.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # Validate
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    running_loss += loss.item() * X_batch.size(0)
                    correct += (outputs.argmax(1) == y_batch).sum().item()
                    total += X_batch.size(0)
            
            val_loss = running_loss / total
            val_acc = correct / total

            print(
                f"Epoch {epoch:>3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
        
        # Test
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        present = sorted(set(all_labels) | set(all_preds))
        target_names = [INDEX_TO_CHAR[i] for i in present]

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            labels=present,
            target_names=target_names,
            zero_division=0,
        )
        
        
        if inplace:
            self.model = model
            self.num_classes = num_classes
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.device = device
            self.accuracy = accuracy
            self.report = report
        
        return model, accuracy, report
    
    
    def predict(self, X):
        input_tensor, _ = _prepare_tensors(X)
        probs = self.model(input_tensor)
        y_pred = probs.argmax(1)
        return y_pred, probs
    
    
    def save_model(self, path):
        """Save full model state dict."""
        os.makedirs(os.path.dirname(path) or self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Model Export] Saved model to path: {path}")


    def load_model(self, path):
        """Load a previously saved model."""
        model = SimpleCNN(self.num_classes).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        print(f"[Model Import] Loaded model from path: {path}")
        return model



class SimpleCNN(nn.Module):
    """
    Compact CNN for 28x28 single-channel character classification.

    Architecture:
        Conv(1->32, 3) -> ReLU -> MaxPool(2)
        Conv(32->64, 3) -> ReLU -> MaxPool(2)
        Conv(64->128, 3) -> ReLU
        Flatten -> FC(512) -> ReLU -> Dropout -> FC(num_classes)

    After two 2x2 max-pools the spatial size is 5x5 (28->13->5 with valid padding
    inside each pool window), giving 128*5*5 = 3200 features before the FC layers.
    """

    def __init__(self, num_classes=36):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

