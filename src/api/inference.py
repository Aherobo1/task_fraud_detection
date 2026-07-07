import os
import sys
import joblib

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.predict import predict_fraud, predict_batch


class FraudDetectionModel:
    """
    Class to handle model loading and inference
    """
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the trained model
        """
        try:
            self.model = joblib.load(config.MODEL_PATH)
            print(f"Model loaded successfully from {config.MODEL_PATH}")
            return True
        except FileNotFoundError:
            print(f"Model file not found at {config.MODEL_PATH}")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, transaction_data):
        """
        Predict fraud for a single transaction
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return predict_fraud(self.model, transaction_data)

    def predict_batch(self, transactions_data):
        """
        Predict fraud for multiple transactions
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return predict_batch(self.model, transactions_data)


# Singleton instance
model_instance = None


def get_model_instance():
    """
    Get or create the model instance
    """
    global model_instance
    if model_instance is None:
        model_instance = FraudDetectionModel()
    return model_instance
