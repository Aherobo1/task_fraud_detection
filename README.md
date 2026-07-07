# Fraud Detection System

## Overview

This project implements a comprehensive fraud detection system that analyzes transaction data, extracts meaningful insights through Exploratory Data Analysis (EDA), performs feature engineering, trains machine learning models to classify fraudulent transactions, and deploys an API with a Web UI for real-time fraud prediction.

The system uses a Random Forest classifier as the core model, achieving high precision and recall in identifying fraudulent transactions. The model is trained on a dataset of credit card transactions with various features including transaction amount, merchant details, cardholder information, and location data.

## Dataset Description

The dataset consists of various features related to transactions, including details about the merchant, transaction amount, user details, and location. The repository now tracks the `data/` folder structure, and the raw CSV files in `data/raw/` are versioned with Git LFS. The key features are:

* **trans_date_trans_time** : Timestamp of the transaction.
* **cc_num** : Credit card number (anonymized transaction number).
* **merchant** : Name of the merchant.
* **category** : Type of merchant.
* **amt** : Amount transferred.
* **first, last** : First and last name of the cardholder.
* **gender** : Gender of the cardholder.
* **street, city, state, zip** : Location details of the cardholder.
* **lat, long** : Latitude and longitude of the cardholder.
* **city_pop** : Population of the city.
* **job** : Job description of the cardholder.
* **dob** : Date of birth of the cardholder.
* **trans_num** : Unique transaction number.
* **unix_time** : Unix timestamp.
* **merch_lat, merch_long** : Latitude and longitude of the merchant.
* **is_fraud** : Target variable (1 for fraud, 0 for legitimate transactions).

## Project Components

### 1. Exploratory Data Analysis (EDA)

The EDA process is documented in the `experiments/eda.ipynb` notebook and includes:

* Analysis of missing values and data distribution
* Visualization of transaction amounts by fraud status
* Correlation analysis between different features
* Geographical patterns of fraudulent transactions
* Identification of high-risk categories and merchants
* Temporal analysis (time of day, day of week) of fraud patterns

### 2. Feature Engineering

Feature engineering is implemented in `src/data_preprocessing.py` and `experiments/feature_engineering.ipynb`, including:

* Extraction of time-based features (hour, day, weekday, month) from transaction timestamps
* Calculation of distance between cardholder and merchant locations
* Derivation of cardholder age from date of birth
* Creation of transaction amount relative to category average
* Handling of categorical variables through one-hot encoding
* Normalization of numerical features

### 3. Model Training

Model training is implemented in `src/model_training.py` and `experiments/model_training.ipynb`, including:

* Data splitting into training and validation sets
* Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
* Training of multiple models (Logistic Regression, Random Forest, Gradient Boosting)
* Hyperparameter optimization
* Model evaluation using accuracy, precision, recall, and F1-score
* Feature importance analysis

### 4. API Implementation

The API is implemented using FastAPI in `src/api/app.py` and provides:

* A `/predict` endpoint for single transaction fraud prediction
* A `/predict/batch` endpoint for batch predictions
* A `/health` endpoint for API status checking
* A `/model-info` endpoint for model metadata

### 5. Web UI

The Web UI is implemented using Flask in `src/web/app.py` and includes:

* A form for entering transaction details
* Real-time fraud prediction display
* Visualization of prediction results
* Model information display

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone http://23.29.118.76:3000/task/task_fraud_detection.git
   cd task_fraud_detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. The raw data files should be placed in the `data/raw/` directory.
2. Run the data preprocessing script to generate the processed data:
   ```bash
   python -m src.data_preprocessing
   ```

### Model Training

1. Train the fraud detection model:
   ```bash
   python -m src.model_training
   ```

2. Evaluate the model performance:
   ```bash
   python -m src.model_evaluation
   ```

### Running the API and Web UI

1. Start the API server:
   ```bash
   python -m src.api.app
   ```

2. In a separate terminal, start the Web UI:
   ```bash
   python -m src.web.app
   ```

3. Access the Web UI in your browser at `http://localhost:8501`

### Using Docker

Alternatively, you can use Docker to run the entire system:

1. Build and start the Docker containers:
   ```bash
   docker-compose -f deployment/docker-compose.yml up --build
   ```

2. Access the Web UI in your browser at `http://localhost:8501`

# Project File Structure:
```
fraud_detection/
│
├── data/                           # Data storage and processing
│   ├── raw/                        # Original dataset files
│   │   ├── fraudTrain.csv          # Training dataset
│   │   └── fraudTest.csv           # Testing dataset
│   └── processed/                  # Processed/cleaned datasets
│       ├── processed_train.csv     # Preprocessed training data
│       ├── processed_test.csv      # Preprocessed testing data
│       └── category_avg.csv        # Category averages for feature engineering
│
├── experiments/                    # Jupyter notebooks for analysis and experimentation
│   ├── eda.ipynb                   # Exploratory Data Analysis notebook
│   ├── feature_engineering.ipynb  # Feature engineering experiments
│   └── model_training.ipynb       # Enhanced model training with comprehensive analysis
│
├── models/                         # Trained models and evaluation artifacts
│   ├── fraud_model.pkl             # Serialized trained RandomForest model
│   ├── model_metadata.json        # Model performance metrics and metadata
│   ├── evaluation_results.json    # Detailed evaluation results
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── feature_importance.png     # Feature importance plot
│   ├── precision_recall_curve.png # Precision-recall curve
│   └── roc_curve.png              # ROC curve visualization
│
├── src/                           # Source code for production system
│   ├── __init__.py                # Python package indicator
│   ├── config.py                  # Configuration settings and paths
│   ├── data_preprocessing.py      # Data cleaning and feature engineering
│   ├── model_training.py          # Model training script
│   ├── model_evaluation.py       # Model evaluation and metrics
│   ├── predict.py                 # Prediction functions and utilities
│   │
│   ├── api/                       # FastAPI backend service
│   │   ├── __init__.py            # Package indicator
│   │   ├── app.py                 # FastAPI application with endpoints
│   │   └── inference.py           # Model loading and inference logic
│   │
│   └── web/                       # Flask web interface
│       ├── __init__.py            # Package indicator
│       ├── app.py                 # Flask web application
│       ├── static/                # Static assets
│       │   ├── css/               # Stylesheets
│       │   └── js/                # JavaScript files
│       └── templates/             # HTML templates
│           ├── index.html         # Main input form
│           ├── result.html        # Prediction results page
│           ├── error.html         # Error handling page
│           └── model_info.html    # Model information display
│
├── deployment/                    # Deployment configurations
│   ├── docker-compose.yml         # Multi-container Docker setup
│   └── cloud_run.sh              # Google Cloud Run deployment script
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container configuration
├── install.sh                     # Installation script
└── checklist.md                   # Development and deployment checklist

```

### Detailed Component Explanation:

#### **📊 Data Pipeline (`data/`)**
* **`raw/`** : Original fraud detection datasets
  * **`fraudTrain.csv`** : Training dataset with transaction records
  * **`fraudTest.csv`** : Testing dataset for model validation
* **`processed/`** : Preprocessed data ready for machine learning
  * **`processed_train.csv`** : Feature-engineered training data
  * **`processed_test.csv`** : Feature-engineered testing data
  * **`category_avg.csv`** : Category averages for transaction normalization

#### **🔬 Experimentation (`experiments/`)**
* **`eda.ipynb`** : Comprehensive exploratory data analysis with visualizations
* **`feature_engineering.ipynb`** : Interactive feature creation and transformation
* **`model_training.ipynb`** : Enhanced training notebook with:
  * Parameter configurations for hypothesis testing
  * Easy model switching between algorithms
  * Detailed confusion matrix analysis
  * Class balancing comparison (SMOTE, downsampling, class weighting)

#### **🤖 Model Artifacts (`models/`)**
* **`fraud_model.pkl`** : Production-ready RandomForest classifier
* **`model_metadata.json`** : Performance metrics and model information
* **`evaluation_results.json`** : Comprehensive evaluation metrics
* **Visualization Files** :
  * **`confusion_matrix.png`** : Model performance visualization
  * **`feature_importance.png`** : Feature importance analysis
  * **`precision_recall_curve.png`** : Precision-recall trade-off
  * **`roc_curve.png`** : ROC curve analysis

#### **💻 Source Code (`src/`)**
* **Core Modules** :
  * **`config.py`** : Centralized configuration and path management
  * **`data_preprocessing.py`** : Data cleaning, feature engineering, and preprocessing pipelines
  * **`model_training.py`** : Model training with hyperparameter optimization
  * **`model_evaluation.py`** : Comprehensive model evaluation and metrics
  * **`predict.py`** : Prediction functions for single and batch processing

* **`api/`** : FastAPI backend service
  * **`app.py`** : REST API with endpoints:
    * `/predict` - Single transaction fraud prediction
    * `/predict/batch` - Batch prediction processing
    * `/health` - Service health monitoring
    * `/model-info` - Model metadata and performance
  * **`inference.py`** : Model loading and prediction logic

* **`web/`** : Flask web interface
  * **`app.py`** : Web application with user-friendly interface
  * **`templates/`** : HTML templates for web pages
    * **`index.html`** : Transaction input form
    * **`result.html`** : Prediction results display
    * **`error.html`** : Error handling page
    * **`model_info.html`** : Model information dashboard
  * **`static/`** : CSS and JavaScript assets for styling and interactivity

#### **🚀 Deployment (`deployment/`)**
* **`docker-compose.yml`** : Multi-container orchestration for API and Web UI
* **`cloud_run.sh`** : Automated Google Cloud Run deployment script

#### **🔧 Development Environment**
* **`requirements.txt`** : Complete list of Python packages and versions
* **`Dockerfile`** : Container configuration for consistent deployment
* **`install.sh`** : Automated setup script for development environment
* **`checklist.md`** : Development progress tracking and deployment checklist
* **`requirements.txt`** : List of Python dependencies.
* **`Dockerfile`** : Container definition for deployment.
* **`deployment/`** : Scripts and configurations for deployment.
  * **`docker-compose.yml`** : Multi-container Docker setup.
  * **`cloud_run.sh`** : Script for deploying to cloud platforms.

## Performance

The Random Forest model achieves the following performance metrics on the validation set:

- **Accuracy**: ~99.84%
- **Precision**: ~94.78% (minimizing false positives)
- **Recall**: ~77.35% (minimizing false negatives)
- **F1 Score**: ~85.18% (balance between precision and recall)

The most important features for fraud detection include:
1. Transaction amount
2. Distance between cardholder and merchant
3. Time of day
4. Transaction category
5. Cardholder age

## Future Improvements

- Implement more advanced models like XGBoost or deep learning
- Add real-time monitoring and alerting capabilities
- Incorporate additional data sources for enhanced fraud detection
- Implement model explainability features
- Add user authentication and authorization to the web interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used in this project is for educational purposes only
- Thanks to all contributors who have helped with the development
