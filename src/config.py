import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Data files
TRAIN_DATA_PATH = RAW_DATA_DIR / 'fraudTrain.csv'
TEST_DATA_PATH = RAW_DATA_DIR / 'fraudTest.csv'

# Processed data files
PROCESSED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / 'processed_train.csv'
PROCESSED_TEST_DATA_PATH = PROCESSED_DATA_DIR / 'processed_test.csv'

# Model files
MODEL_PATH = MODELS_DIR / 'fraud_model.pkl'
MODEL_METADATA_PATH = MODELS_DIR / 'model_metadata.json'

# API settings
API_HOST = '0.0.0.0'
API_PORT = 8001

# Web UI settings
WEB_HOST = '0.0.0.0'
WEB_PORT = 8501
