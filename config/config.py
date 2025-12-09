import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    
    # Model files
    MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')
    COLUMNS_PATH = os.path.join(MODEL_DIR, 'columns.pkl')  # Store column names
    
    # Training parameters
    TEST_SIZE = 0.30
    RANDOM_STATE = 42
    
    # Feature columns
    NUMERICAL_COLS = ['tenure', 'Monthly_Charges', 'Total_Charges']
    
    # Model parameters
    GRADIENT_BOOSTING_PARAMS = {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    }
    
    LOGISTIC_REGRESSION_PARAMS = {
        'random_state': RANDOM_STATE,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    
    ADABOOST_PARAMS = {
        'random_state': RANDOM_STATE,
        'n_estimators': 50
    }
    
    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)