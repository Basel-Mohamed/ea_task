import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import ChurnPredictor
from src.utils import format_prediction_response
import json


def predict_single_customer():
    # \"\"\"Predict for a single customer\"\"\"
    # Load model
    predictor = ChurnPredictor()
    predictor.load()
    
    # Sample customer data
    sample_customer = {
        'gender': 'Female',
        'Senior_Citizen': 0,  
        'Is_Married': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'Phone_Service': 'No',
        'Dual': 'No',
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
    
    # Predict
    result = predictor.predict_single(sample_customer)
    
    # Format and print
    response = format_prediction_response(result)
    print(json.dumps(response, indent=2))
    
    return result


if __name__ == "__main__":
    print("Running single customer prediction...")
    predict_single_customer()