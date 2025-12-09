import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union
import sys
sys.path.append('..')
from config.config import Config
from src.data_processor import DataProcessor
from src.model import ChurnModel


class ChurnPredictor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = ChurnModel()
        self.config = Config
        self.is_trained = False
    
    def train_pipeline(self, data_path: str):
        """Complete training pipeline"""
        print("Starting training pipeline...")
        
        # Load and prepare data
        print("Loading data...")
        df = self.data_processor.load_data(data_path)
        
        print("Preparing data...")
        X, y = self.data_processor.prepare_data(df, is_training=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Evaluate
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        metrics = self.model.evaluate(X_test, y_test)
        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        self.is_trained = True
        return metrics
    
    def save(self, model_path: str = None, scaler_path: str = None, 
             encoders_path: str = None, columns_path: str = None):
        """Save model and all preprocessing objects"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Train before saving.")
        
        model_path = model_path or self.config.MODEL_PATH
        scaler_path = scaler_path or self.config.SCALER_PATH
        encoders_path = encoders_path or self.config.ENCODERS_PATH
        columns_path = columns_path or self.config.COLUMNS_PATH
        
        self.config.ensure_directories()
        
        joblib.dump(self.model.model, model_path)
        joblib.dump(self.data_processor.scaler, scaler_path)
        joblib.dump(self.data_processor.label_encoders, encoders_path)
        joblib.dump(self.data_processor.feature_columns, columns_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Encoders saved to: {encoders_path}")
        print(f"Columns saved to: {columns_path}")
    
    def load(self, model_path: str = None, scaler_path: str = None, 
             encoders_path: str = None, columns_path: str = None):
        """Load model and all preprocessing objects"""
        model_path = model_path or self.config.MODEL_PATH
        scaler_path = scaler_path or self.config.SCALER_PATH
        encoders_path = encoders_path or self.config.ENCODERS_PATH
        columns_path = columns_path or self.config.COLUMNS_PATH
        
        self.model.model = joblib.load(model_path)
        self.data_processor.scaler = joblib.load(scaler_path)
        self.data_processor.label_encoders = joblib.load(encoders_path)
        self.data_processor.feature_columns = joblib.load(columns_path)
        
        self.is_trained = True
        print("Model and preprocessors loaded successfully")
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """Predict churn for a single customer"""
        if not self.is_trained:
            raise ValueError("Model not loaded. Load or train model first.")
        
        # Store customer ID before processing
        customer_id = customer_data.get('customerID', customer_data.get('customer_id', 'N/A'))
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Prepare data (this will remove customerID)
        X_prepared, _ = self.data_processor.prepare_data(
            df, 
            is_training=False
        )
        
        # Predict
        prediction = self.model.predict(X_prepared)[0]
        probabilities = self.model.predict_proba(X_prepared)[0]
        
        return {
            'customerID': customer_id,
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': float(probabilities[1]),
            'retention_probability': float(probabilities[0]),
            'risk_level': self._get_risk_level(probabilities[1])
        }
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Categorize churn risk"""
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'

# Import for metrics
from sklearn.metrics import accuracy_score