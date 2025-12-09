# Telco Customer Churn Prediction

A production-ready machine learning system for predicting customer churn using ensemble methods.

## Project Structure

```
churn_prediction/
├── config/              # Configuration files
├── data/               # Data directory
├── models/             # Saved models
├── src/                # Source code
├── api/                # Flask API
├── tests/              # Unit tests
├── train.py            # Training script
├── inference.py        # Inference script
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Inference

Single prediction:
```bash
python inference.py
```


### API

Start the API server:
```bash
python api/app.py
```

Endpoints:
- `GET /health` - Health check
- `POST /predict` - Single prediction

Example request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Senior_Citizen": 0,
    "tenure": 12,
    ...
  }'
```

## Features

- Clean, modular architecture
- Separate data processing and modeling
- Easy API integration
- Batch and single predictions
- Risk level classification
- Comprehensive error handling

## Model Performance

The ensemble model combines:
- Gradient Boosting
- Logistic Regression
- AdaBoost

Using soft voting for optimal predictions.