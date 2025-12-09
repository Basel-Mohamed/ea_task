"""
Telco Customer Churn Prediction
A clean, modular approach to predicting customer churn using ensemble methods
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


class ChurnPredictor:
    """Main class for customer churn prediction"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.numerical_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Drop customer ID
        df = df.drop(['customerID'], axis=1)
        
        # Convert Total_Charges to numeric
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
        df['Total_Charges'].fillna(df['Total_Charges'].mean(), inplace=True)
        
        # Remove zero tenure records
        df = df[df['tenure'] != 0]
        
        return df
    
    def encode_features(self, df, is_training=True):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def prepare_features(self, X, is_training=True):
        """Standardize numerical features"""
        X_prepared = X.copy()
        
        if is_training:
            X_prepared[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        else:
            X_prepared[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        
        return X_prepared
    
    def build_ensemble_model(self):
        """Build voting classifier with ensemble of models"""
        clf1 = GradientBoostingClassifier(random_state=42)
        clf2 = LogisticRegression(random_state=42, max_iter=1000)
        clf3 = AdaBoostClassifier(random_state=42)
        
        ensemble = VotingClassifier(
            estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)],
            voting='soft'
        )
        
        return ensemble
    
    def train(self, test_size=0.30, random_state=42):
        """Complete training pipeline"""
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess_data()
        
        print("Encoding features...")
        df_encoded = self.encode_features(df, is_training=True)
        
        # Split features and target
        X = df_encoded.drop(columns=['Churn'])
        y = df_encoded['Churn'].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print("Preparing features...")
        X_train = self.prepare_features(X_train, is_training=True)
        X_test = self.prepare_features(X_test, is_training=False)
        
        # Build and train model
        print("Training ensemble model...")
        self.model = self.build_ensemble_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nModel Performance:")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Testing Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        return self.model
    
    def save_model(self, model_path='churn_model.pkl', 
                   scaler_path='scaler.pkl', 
                   encoders_path='encoders.pkl'):
        """Save trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Encoders saved to {encoders_path}")
    
    def load_model(self, model_path='churn_model.pkl', 
                   scaler_path='scaler.pkl', 
                   encoders_path='encoders.pkl'):
        """Load trained model and preprocessors"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        print("Model and preprocessors loaded successfully")
    
    def predict_single(self, customer_data):
        """
        Predict churn for a single customer
        
        Parameters:
        -----------
        customer_data : dict
            Dictionary containing customer features
            
        Returns:
        --------
        dict: Prediction results with probability
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        
        # Convert to DataFrame
        df_input = pd.DataFrame([customer_data])
        
        # Encode features
        df_encoded = self.encode_features(df_input, is_training=False)
        
        # Prepare features
        df_prepared = self.prepare_features(df_encoded, is_training=False)
        
        # Predict
        prediction = self.model.predict(df_prepared)[0]
        probability = self.model.predict_proba(df_prepared)[0]
        
        return {
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': probability[1],
            'retention_probability': probability[0]
        }
    
    def predict_batch(self, data_path):
        """Predict churn for multiple customers from CSV"""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Store customer IDs if present
        customer_ids = df['customerID'] if 'customerID' in df.columns else None
        
        # Remove unnecessary columns
        df = df.drop(['customerID'], axis=1, errors='ignore')
        if 'Churn' in df.columns:
            df = df.drop(['Churn'], axis=1)
        
        # Preprocess
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
        df['Total_Charges'].fillna(df['Total_Charges'].mean(), inplace=True)
        
        # Encode and prepare
        df_encoded = self.encode_features(df, is_training=False)
        df_prepared = self.prepare_features(df_encoded, is_training=False)
        
        # Predict
        predictions = self.model.predict(df_prepared)
        probabilities = self.model.predict_proba(df_prepared)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_prediction': ['Yes' if p == 1 else 'No' for p in predictions],
            'churn_probability': probabilities[:, 1],
            'retention_probability': probabilities[:, 0]
        })
        
        if customer_ids is not None:
            results.insert(0, 'customerID', customer_ids)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor(r'D:\Neurotech\credit card\Etisalat_UseCase\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Train model
    predictor.train()
    
    # Save model
    predictor.save_model()
    
    # Example: Inference on single customer
    sample_customer = {
        'gender': 'Female',
        'Senior_Citizen ': 0,
        'Is_Married': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'Phone_Service': 'No',
        'Dual': 'No',  # Changed from Multiple_Lines
        'Internet_Service': 'Fiber optic',
        'Online_Security': 'No',
        'Online_Backup': 'No',
        'Device_Protection': 'No',
        'Tech_Support': 'No',
        'Streaming_TV': 'Yes',
        'Streaming_Movies': 'Yes',
        'Contract': 'Month-to-month',
        'Paperless_Billing': 'Yes',
        'Payment_Method': 'Electronic check',
        'Monthly_Charges': 70.35,
        'Total_Charges': 844.2
    }
    
    result = predictor.predict_single(sample_customer)
    print("\nSingle Customer Prediction:")
    print(f"Churn Prediction: {result['churn_prediction']}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Retention Probability: {result['retention_probability']:.2%}")