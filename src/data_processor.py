import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import sys
sys.path.append('..')
from config.config import Config


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = Config.NUMERICAL_COLS
        self.feature_columns = None  # Store original column names
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by stripping whitespace"""
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip()
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and handle missing values"""
        df_clean = df.copy()
        
        # Clean column names first
        df_clean = self.clean_column_names(df_clean)
        
        # Drop customer ID if exists
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(['customerID'], axis=1)
        
        # Convert Total_Charges to numeric
        df_clean['Total_Charges'] = pd.to_numeric(
            df_clean['Total_Charges'], 
            errors='coerce'
        )
        # Use loc to avoid Warning
        mean_charges = df_clean['Total_Charges'].mean()
        df_clean.loc[:, 'Total_Charges'] = df_clean['Total_Charges'].fillna(mean_charges)
        
        # Remove zero tenure records
        df_clean = df_clean[df_clean['tenure'] != 0].copy()
        
        return df_clean
    
    def encode_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Clean column names
        df_encoded = self.clean_column_names(df_encoded)
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col]
                    )
                else:
                    if col not in self.label_encoders:
                        raise ValueError(f"No encoder found for column: {col}")
                    df_encoded[col] = self.label_encoders[col].transform(
                        df_encoded[col]
                    )
        
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        X_scaled = X.copy()
        
        if is_training:
            X_scaled[self.numerical_cols] = self.scaler.fit_transform(
                X[self.numerical_cols]
            )
        else:
            X_scaled[self.numerical_cols] = self.scaler.transform(
                X[self.numerical_cols]
            )
        
        return X_scaled
    
    def align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has the same columns as training data"""
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Train the model first.")
        
        df_aligned = df.copy()
        
        # Add missing columns with 0
        for col in self.feature_columns:
            if col not in df_aligned.columns:
                df_aligned[col] = 0
        
        # Remove extra columns
        df_aligned = df_aligned[self.feature_columns]
        
        return df_aligned
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        is_training: bool = True,
        target_col: str = 'Churn'
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Complete data preparation pipeline"""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Separate features and target
        if target_col in df_clean.columns:
            y = df_clean[target_col].copy()
            X = df_clean.drop(columns=[target_col])
        else:
            y = None
            X = df_clean.copy()
        
        # Encode categorical features
        X_encoded = self.encode_features(X, is_training=is_training)
        
        # Store feature columns during training
        if is_training:
            self.feature_columns = X_encoded.columns.tolist()
        else:
            # Align columns for inference
            X_encoded = self.align_columns(X_encoded)
        
        # Scale numerical features
        X_prepared = self.scale_features(X_encoded, is_training=is_training)
        
        # Encode target if present
        if y is not None and is_training:
            if y.dtype == 'object':
                if target_col not in self.label_encoders:
                    self.label_encoders[target_col] = LabelEncoder()
                y = self.label_encoders[target_col].fit_transform(y)
        
        return X_prepared, y