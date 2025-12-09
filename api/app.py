from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import ChurnPredictor
from src.utils import validate_input_data, format_prediction_response

from .chatbot_service import ChurnChatbotService
import uuid

app = Flask(__name__)
CORS(app)

# Load model at startup
predictor = ChurnPredictor()
# Initialize chatbot service
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
chatbot_service = ChurnChatbotService(GROQ_API_KEY)

conversation_sessions = {}

try:
    predictor.load()
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    # \"\"\"Health check endpoint\"\"\"
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_trained
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    # \"\"\"Single customer prediction endpoint\"\"\"
    try:
        # Get data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validate input
        validate_input_data(data)
        
        # Make prediction
        result = predictor.predict_single(data)
        
        # Format response
        response = format_prediction_response(result)
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        # Print full traceback for debugging
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint for conversational churn prediction
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No message provided'
            }), 400
        
        user_message = data['message']
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Get or create conversation session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ChurnChatbotService(GROQ_API_KEY)
        
        chatbot = conversation_sessions[session_id]
        
        # Process message
        result = chatbot.chat(user_message)
        
        # If action is predict, call the prediction API
        if result['action'] == 'predict' and result['data']:
            try:
                prediction_result = predictor.predict_single(result['data'])
                
                # Format friendly response
                churn_prob = prediction_result['churn_probability']
                risk_level = prediction_result['risk_level']
                prediction = prediction_result['churn_prediction']
                customer_id = prediction_result.get('customerID', 'N/A')
                
                friendly_response = f"""
📊 Churn Prediction Results for Customer {customer_id}

🎯 Prediction: {prediction}
📈 Churn Probability: {churn_prob:.1%}
⚠️ Risk Level: {risk_level}

"""
                
                if prediction == 'Yes':
                    if risk_level == 'High':
                        friendly_response += "🔴 Alert: This customer has a HIGH risk of churning! Immediate retention actions recommended.\n\n"
                        friendly_response += "💡 Recommendations:\n"
                        friendly_response += "- Offer personalized retention incentives\n"
                        friendly_response += "- Schedule a call from customer success team\n"
                        friendly_response += "- Consider contract upgrade with benefits"
                    elif risk_level == 'Medium':
                        friendly_response += "🟡 Caution: This customer shows moderate churn risk.\n\n"
                        friendly_response += "💡 Recommendations:\n"
                        friendly_response += "- Monitor account activity closely\n"
                        friendly_response += "- Send satisfaction survey\n"
                        friendly_response += "- Highlight value-added services"
                else:
                    friendly_response += "🟢 Good News: This customer is likely to stay with us!\n\n"
                    friendly_response += "💡 Recommendations:\n"
                    friendly_response += "- Continue excellent service\n"
                    friendly_response += "- Consider upsell opportunities\n"
                    friendly_response += "- Request referrals"
                
                friendly_response += "\n\nWould you like to check another customer?"
                
                return jsonify({
                    'status': 'success',
                    'response': friendly_response,
                    'session_id': session_id,
                    'prediction_data': prediction_result
                }), 200
                
            except Exception as pred_error:
                return jsonify({
                    'status': 'error',
                    'response': f"I gathered the information, but encountered an error making the prediction: {str(pred_error)}",
                    'session_id': session_id
                }), 500
        
        # Return chat response
        return jsonify({
            'status': 'success',
            'response': result['response'],
            'session_id': session_id
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    """Reset conversation for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in conversation_sessions:
            conversation_sessions[session_id].reset_conversation()
            return jsonify({
                'status': 'success',
                'message': 'Conversation reset successfully'
            }), 200
        
        return jsonify({
            'status': 'success',
            'message': 'No active conversation to reset'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
        
if __name__ == '__main__':
    # Development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # Gunicorn will handle the app
    pass
