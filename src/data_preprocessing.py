import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# datetime is used implicitly when working with pandas datetime objects
from geopy.distance import geodesic
import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def load_data(file_path):
    """
    Load data from CSV file
    """
    return pd.read_csv(file_path)


def calculate_distance(row):
    """
    Calculate the distance between the cardholder and merchant in kilometers
    """
    try:
        cardholder_coords = (row['lat'], row['long'])
        merchant_coords = (row['merch_lat'], row['merch_long'])
        return geodesic(cardholder_coords, merchant_coords).kilometers
    except:
        return np.nan


def extract_time_features(df):
    """
    Extract time-based features from the transaction timestamp
    """
    # Convert to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Extract features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['weekday'] = df['trans_date_trans_time'].dt.weekday
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year

    # Create is_weekend feature
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Create time of day categories
    df['time_of_day'] = df['hour'].apply(lambda x:
                                        'night' if 0 <= x < 6 else
                                        'morning' if 6 <= x < 12 else
                                        'afternoon' if 12 <= x < 18 else
                                        'evening')

    return df


def calculate_age(df):
    """
    Calculate age of the cardholder based on date of birth
    """
    # Convert to datetime
    df['dob'] = pd.to_datetime(df['dob'])

    # Calculate age at the time of transaction
    df['age'] = df.apply(lambda row: (row['trans_date_trans_time'].year - row['dob'].year) -
                        ((row['trans_date_trans_time'].month, row['trans_date_trans_time'].day) <
                         (row['dob'].month, row['dob'].day)), axis=1)

    return df


def preprocess_data(df, is_training=True):
    """
    Preprocess the data for model training or prediction
    """
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()

    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Fix for FutureWarning - avoid chained assignment with inplace=True
            df_processed[col] = df_processed[col].fillna('unknown')
        else:
            # Fix for FutureWarning - avoid chained assignment with inplace=True
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    # Extract time features
    df_processed = extract_time_features(df_processed)

    # Calculate age
    df_processed = calculate_age(df_processed)

    # Calculate distance between cardholder and merchant
    df_processed['distance_km'] = df_processed.apply(calculate_distance, axis=1)

    # Create feature for transaction amount relative to average for that category
    if is_training:
        category_avg = df_processed.groupby('category')['amt'].mean().to_dict()
    else:
        # Load the category averages from the training data
        # This would be stored during training
        category_avg = pd.read_csv(config.PROCESSED_DATA_DIR / 'category_avg.csv').set_index('category')['amt'].to_dict()

    df_processed['amt_to_category_avg'] = df_processed.apply(
        lambda row: row['amt'] / category_avg.get(row['category'], 1), axis=1)

    # Select features for model
    feature_cols = [
        'amt', 'distance_km', 'age', 'hour', 'day', 'weekday', 'month',
        'is_weekend', 'amt_to_category_avg', 'city_pop', 'category', 'time_of_day'
    ]

    # For training data, save the category averages
    if is_training:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        pd.DataFrame(list(category_avg.items()), columns=['category', 'amt']).to_csv(
            config.PROCESSED_DATA_DIR / 'category_avg.csv', index=False)

    # Return the processed data with selected features
    return df_processed[feature_cols + (['is_fraud'] if 'is_fraud' in df_processed.columns else [])]


def get_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for numerical and categorical features
    """
    # Define numerical and categorical features
    numerical_features = [
        'amt', 'distance_km', 'age', 'hour', 'day', 'weekday', 'month',
        'amt_to_category_avg', 'city_pop'
    ]

    categorical_features = ['category', 'time_of_day']
    binary_features = ['is_weekend']

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])

    return preprocessor


def main():
    """
    Main function to preprocess the data and save it
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    print("Loading training data...")
    train_data = load_data(config.TRAIN_DATA_PATH)

    print("Loading test data...")
    test_data = load_data(config.TEST_DATA_PATH)

    print("Preprocessing training data...")
    processed_train = preprocess_data(train_data, is_training=True)

    print("Preprocessing test data...")
    processed_test = preprocess_data(test_data, is_training=False)

    print("Saving processed data...")
    processed_train.to_csv(config.PROCESSED_TRAIN_DATA_PATH, index=False)
    processed_test.to_csv(config.PROCESSED_TEST_DATA_PATH, index=False)

    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()
