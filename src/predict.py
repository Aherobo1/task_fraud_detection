import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.data_preprocessing import preprocess_data


def load_model():
    """
    Load the trained model
    """
    try:
        model = joblib.load(config.MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {config.MODEL_PATH}")
        return None


def load_model_metadata():
    """
    Load the model metadata
    """
    try:
        with open(config.MODEL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        print(f"Model metadata file not found at {config.MODEL_METADATA_PATH}")
        return None


def prepare_input_data(transaction_data):
    """
    Prepare input data for prediction
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(transaction_data, dict):
        transaction_data = pd.DataFrame([transaction_data])

    # Preprocess the data
    processed_data = preprocess_data(transaction_data, is_training=False)

    return processed_data


def predict_fraud(model, transaction_data):
    """
    Predict fraud for a transaction
    """
    # Prepare the input data
    processed_data = prepare_input_data(transaction_data)

    # Make prediction
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0, 1]  # Probability of fraud

    result = {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'risk_level': 'high' if probability > 0.7 else 'medium' if probability > 0.3 else 'low'
    }

    return result


def predict_batch(model, transactions_data):
    """
    Predict fraud for multiple transactions
    """
    # Prepare the input data
    processed_data = prepare_input_data(transactions_data)

    # Make predictions
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]  # Probabilities of fraud

    results = []
    for i in range(len(predictions)):
        result = {
            'is_fraud': bool(predictions[i]),
            'fraud_probability': float(probabilities[i]),
            'risk_level': 'high' if probabilities[i] > 0.7 else 'medium' if probabilities[i] > 0.3 else 'low'
        }
        results.append(result)

    return results


def main():
    """
    Main function for demonstration
    """
    # Load the model
    print("Loading the model...")
    model = load_model()
    if model is None:
        return

    # Example transaction data
    example_transaction = {
        'trans_date_trans_time': '2019-01-01 00:00:00',
        'cc_num': '4532315247148429',
        'merchant': 'fraud_Rippin, Kub and Mann',
        'category': 'grocery_pos',
        'amt': 4.97,
        'first': 'John',
        'last': 'Doe',
        'gender': 'M',
        'street': '123 Main St',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 8336817,
        'job': 'Data Scientist',
        'dob': '1980-01-01',
        'trans_num': 'a795d3a0f8f11f9c45d3a4aa62b5c0f3',
        'unix_time': 1546300800,
        'merch_lat': 40.7128,
        'merch_long': -74.0060
    }

    # Make prediction
    print("Making prediction...")
    result = predict_fraud(model, example_transaction)

    # Print result
    print("\nPrediction Result:")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")


if __name__ == "__main__":
    main()
