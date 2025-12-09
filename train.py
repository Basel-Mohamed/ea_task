import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import ChurnPredictor
from config.config import Config


def main():
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Path to training data
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Train model
    predictor.train_pipeline(data_path)
    
    # Save model
    predictor.save()
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()