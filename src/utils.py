import json
import pandas as pd
from typing import Dict, Any


def validate_input_data(data: Dict) -> bool:
    # \"\"\"Validate input data has all required fields\"\"\"
    # Get the actual column names from the data
    required_fields = list(data.keys())
    
    # Basic validation - check for essential fields
    essential_fields = ['tenure', 'Monthly_Charges', 'Total_Charges']
    
    missing_fields = [field for field in essential_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    return True


def format_prediction_response(prediction: Dict) -> Dict:
    # \"\"\"Format prediction response\"\"\"
    response = {
        'status': 'success',
        'data': prediction,
        'message': 'Prediction completed successfully'
    }
    return response


def save_predictions_to_csv(predictions: pd.DataFrame, output_path: str):
    # \"\"\"Save predictions to CSV\"\"\"
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def load_json_config(config_path: str) -> Dict[str, Any]:
    # \"\"\"Load JSON configuration file\"\"\"
    with open(config_path, 'r') as f:
        return json.load(f)