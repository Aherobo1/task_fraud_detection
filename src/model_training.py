import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.data_preprocessing import get_preprocessing_pipeline


def train_model(X_train, y_train, X_val=None, y_val=None, use_smote=True):
    """
    Train a model on the given data
    """
    # Get preprocessing pipeline
    preprocessor = get_preprocessing_pipeline()

    # Handle categorical features before SMOTE
    print("Preprocessing data...")
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")

    if use_smote and categorical_cols:
        # We need to preprocess categorical features before applying SMOTE
        print("Preprocessing categorical features for SMOTE...")
        # Create a preprocessing pipeline just for categorical features
        cat_preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )

        # Apply preprocessing to training data
        X_train_processed = cat_preprocessor.fit_transform(X_train)

        # Apply SMOTE to the preprocessed data
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

        # For the final pipeline, we'll use the original data and let the full preprocessor handle it
        X_train_for_pipeline, y_train_for_pipeline = X_train, y_train
    elif use_smote:
        # If no categorical features, apply SMOTE directly
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train_for_pipeline, y_train_for_pipeline = X_train_resampled, y_train_resampled
    else:
        # No SMOTE, use original data
        X_train_for_pipeline, y_train_for_pipeline = X_train, y_train

    # Create the full pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameters for grid search
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Perform grid search with cross-validation
    print("Performing grid search with cross-validation...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_for_pipeline, y_train_for_pipeline)

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        y_pred = best_model.predict(X_val)
        print("Validation metrics:")
        print_metrics(y_val, y_pred)

    return best_model, grid_search.best_params_


def print_metrics(y_true, y_pred):
    """
    Print evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance
    """
    # Get feature importance from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = model.named_steps['classifier'].feature_importances_

    # Get the transformed feature names from the pipeline
    if hasattr(model, 'named_steps'):
        # For pipeline models, get the feature names from the preprocessor
        preprocessor = model.named_steps['preprocessor']
        # Get the transformed feature names
        transformed_features = []

        # Handle numerical features (they keep their names)
        numerical_features = preprocessor.transformers_[0][2]  # Numerical features list
        transformed_features.extend(numerical_features)

        # Handle categorical features (they get expanded with one-hot encoding)
        categorical_features = preprocessor.transformers_[1][2]  # Categorical features list
        categorical_transformer = preprocessor.transformers_[1][1]  # OneHotEncoder
        if hasattr(categorical_transformer, 'get_feature_names_out'):
            # For newer scikit-learn versions
            cat_feature_names = categorical_transformer.get_feature_names_out(categorical_features)
        else:
            # For older scikit-learn versions
            cat_feature_names = categorical_transformer.named_steps['onehot'].get_feature_names(categorical_features)
        transformed_features.extend(cat_feature_names)

        # Handle binary features (they pass through)
        binary_features = preprocessor.transformers_[2][2]  # Binary features list
        transformed_features.extend(binary_features)

        # Use the transformed feature names
        feature_names = transformed_features

    # Make sure the lengths match
    if len(feature_names) != len(importances):
        print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")
        # Use generic feature names if lengths don't match
        feature_names = [f'Feature {i}' for i in range(len(importances))]

    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODELS_DIR, 'feature_importance.png'))
    plt.close()

    return feature_importance


def save_model(model, metadata):
    """
    Save the trained model and its metadata
    """
    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Save the model
    joblib.dump(model, config.MODEL_PATH)

    # Save metadata
    with open(config.MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Model metadata saved to {config.MODEL_METADATA_PATH}")


def main():
    """
    Main function to train the model
    """
    # Load processed data
    print("Loading processed training data...")
    try:
        train_data = pd.read_csv(config.PROCESSED_TRAIN_DATA_PATH)
    except FileNotFoundError:
        print("Processed training data not found. Please run data_preprocessing.py first.")
        return

    # Split features and target
    X = train_data.drop('is_fraud', axis=1)
    y = train_data['is_fraud']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Train the model
    print("Training the model...")
    model, best_params = train_model(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    y_pred = model.predict(X_val)
    metrics = print_metrics(y_val, y_pred)

    # Get feature names after preprocessing
    feature_names = X.columns.tolist()

    # Plot feature importance
    print("\nPlotting feature importance...")
    feature_importance = plot_feature_importance(model, feature_names)

    # Save the model and metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'best_parameters': best_params,
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict(orient='records'),
        'features': feature_names
    }

    save_model(model, metadata)

    print("Model training completed!")


if __name__ == "__main__":
    main()
