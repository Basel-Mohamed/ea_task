# e& Customer Churn Prediction & Chatbot API
Base URL: ```http://127.0.0.1:5000``` (Local) / ```https://eatask-production.up.railway.app``` (Production)

This API provides endpoints for:

1- Direct Churn Prediction: Submit raw customer data to get an instant risk assessment.

2- Conversational Chatbot: An intelligent agent that collects customer data through natural conversation and triggers predictions automatically.

3- System Health: Monitoring the service status and model loading state.

## 1. Health Check
Verifies if the API server is running and the machine learning model is loaded into memory.

Endpoint: ```/health```

Method: ```GET```

### Response:

```JSON
{
  "status": "healthy",
  "model_loaded": true
}
```

## 2. Predict Customer Churn (Direct)
Directly predicts churn probability and risk level for a single customer based on structured JSON input.

Endpoint: ```/predict```

Method: ```POST``

Headers: ```Content-Type: application/json```

### Example Request

```JSON

{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "Senior_Citizen": 0,
    "Is_Married": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Phone_Service": "No",
    "Dual": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "No",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "Yes",
    "Streaming_Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Monthly_Charges": 70.35,
    "Total_Charges": 844.2
}
```

### Success Response

```JSON

{
    "churn_prediction": "Yes",
    "churn_probability": 0.784,
    "risk_level": "High",
    "customerID": "7590-VHVEG"
}

```
## 3. Chat with Agent
Interact with the AI agent to gather customer information conversationally. The agent will automatically trigger a prediction once enough data is collected.

Endpoint: ```/chat```

Method: ```POST```

Headers: `Content-Type: application/json`

### Example Request
```JSON

{
  "message": "I have a customer who wants to cancel. She has a month-to-month contract.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
### Response (Ongoing Conversation)

```JSON

{
  "status": "success",
  "response": "I can help with that. Could you tell me her tenure and what kind of internet service she has?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Response (Prediction Triggered)
When the AI has gathered all required data, it returns the prediction results:

```JSON

{
  "status": "success",
  "response": "\n📊 Churn Prediction Results for Customer 7590-VHVEG\n\n🎯 Prediction: Yes\n📈 Churn Probability: 78.4%\n⚠️ Risk Level: High\n\n🔴 Alert: This customer has a HIGH risk of churning! Immediate retention actions recommended.\n\n💡 Recommendations:\n- Offer personalized retention incentives\n- Schedule a call from customer success team\n\nWould you like to check another customer?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction_data": {
      "churn_probability": 0.784,
      "risk_level": "High",
      "churn_prediction": "Yes",
      "customerID": "7590-VHVEG"
  }
}
  ```


## 4. Reset Chat Session
Clears the conversation history for a specific session ID, allowing you to start fresh with a new customer.

Endpoint: `/chat/reset`

Method: `POST`

Headers: `Content-Type: application/json`

### Request Body

```JSON

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
### Success Response
```JSON

{
  "status": "success",
  "message": "Conversation reset successfully"
}
```