from sklearn.ensemble import (
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import numpy as np
import sys
sys.path.append('..')
from config.config import Config


class ChurnModel:
    def __init__(self):
        self.model = None
        self.config = Config
    
    def build_model(self) -> VotingClassifier:
        """Build ensemble model"""
        clf1 = GradientBoostingClassifier(**self.config.GRADIENT_BOOSTING_PARAMS)
        clf2 = LogisticRegression(**self.config.LOGISTIC_REGRESSION_PARAMS)
        clf3 = AdaBoostClassifier(**self.config.ADABOOST_PARAMS)
        
        ensemble = VotingClassifier(
            estimators=[
                ('gbc', clf1), 
                ('lr', clf2), 
                ('abc', clf3)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        print("Building ensemble model...")
        self.model = self.build_model()
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)