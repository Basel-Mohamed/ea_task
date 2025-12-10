# 🔮 Customer Churn Prediction & AI Assistant

An end-to-end AI solution that combines **Classical Machine Learning** with **Large Language Models (LLM)** to predict customer churn. This project features a robust training pipeline, a Flask REST API, and a conversational chatbot interface that allows marketing teams to assess customer risk using natural language.

## 📌 Project Overview

This system helps telecom retention teams identify at-risk customers early and suggests actionable strategies to prevent churn.

* **Problem:** Predicting whether a customer will leave (`Churn`) based on demographics, services, and billing behavior.
* **Solution:** An **Ensemble Voting Classifier** (Gradient Boosting + Logistic Regression + AdaBoost) for high stability and generalization.
* **Key Innovation:** An **LLM-powered Agent** (via Groq API) that acts as a friendly interface, gathering customer details conversationally before triggering the prediction model.
* **Performance:** The model achieves:
    * **Accuracy:** ~82%
    * **Retention Recall:** ~90% 
    * **F1-Score:** ~87%

---

# 📂 Project Structure

```
churn_prediction/
├── config/              # Configuration files
├── data/               # Data directory
├── models/             # Saved models
├── src/                # Source code
├── api/                # Flask API
├── train.py            # Training script
├── inference.py        # Inference script
└── requirements.txt    # Dependencies
```


## you can use the prduction URL direct:
for Backend `https://eatask-production.up.railway.app/`

for Frontend `https://ea-task-frontend.vercel.app/`

# 🚀 Setup & Installation
## 1. Prerequisites
Python 3.9+

Groq API Key (for the chatbot feature)

## 2. Clone the Repository
Bash

```bash
git clone https://github.com/Basel-Mohamed/ea_task.git

cd ea-Task
```
## 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
## 4. Install Dependencies

```bash
pip install -r requirements.txt
```
# 🛠️ How to Run

## Step 1: Train the Model

Before running the API, you must train the model to generate the necessary artifacts (.pkl files) in the models/ directory.

```bash
python train.py
```
* This script will:
* Load the data
* Clean it
* Engineer features
* Train the Random Forest model
* Save the trained artifacts


## Step 2: Configure Environment: Create a `.env` file in the root directory and add your API key:

```bash
GROQ_API_KEY=your_api_key_here
```

## Step 3: Start the API Server

Once training is complete, start the FastAPI server:

```bash
gunicorn api.app:app
```
The server will start at:
```bash
http://127.0.0.1:8000

```

# 🔌 API Endpoints

1. Health Check

URL: ```GET /health```
Description: Checks if the API is running and the model is loaded.

2. Predict Sales

URL: ```POST /predict```
Description: Returns churn probability and risk level for raw structured data.

Request Body (JSON)

c
```json
{
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

3. Chat Agent

URL: ```POST /chat```
Description: Interacts with the LLM to gather data and trigger predictions conversationally.

4. Clear History

URL: ```POST /chat/reset```
Description: Clears the conversation history for a specific session ID, allowing you to start fresh with a new customer.

For Example
```JSON

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## 📊 Key Features
### ✅ Conversational AI
Smart extraction of customer details from natural text.

Asks clarification questions for missing data.

Generates business-friendly risk reports.

### ✅ Advanced Ensemble Modeling
Combines Gradient Boosting, Logistic Regression, and AdaBoost.

Uses Soft Voting for calibrated probability estimation.

### ✅ Robust Engineering
Label Encoding: Handles categorical variables automatically.

Scaling: Standardizes numerical features for model stability.

Modular Code: Separated logic for data processing, training, and inference.

## 🚀 Future Roadmap
* On-Premises LLM Deployment: Move from Groq API to a self-hosted Ollama instance if data privacy requirements become stricter and GPU infrastructure becomes available.

* Database Integration: Migrate to PostgreSQL to store long-term customer history.

* Model Monitoring: Implement a dashboard to track prediction accuracy over time and detect data drift.

* Automated Retraining: Implement MLOps pipelines (e.g., using Airflow) to retrain the model monthly on fresh data.

