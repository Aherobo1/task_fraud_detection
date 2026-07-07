import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


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


def load_test_data():
    """
    Load the processed test data
    """
    try:
        test_data = pd.read_csv(config.PROCESSED_TEST_DATA_PATH)
        return test_data
    except FileNotFoundError:
        print(f"Test data file not found at {config.PROCESSED_TEST_DATA_PATH}")
        return None


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data
    """
    if 'is_fraud' not in test_data.columns:
        print("Target variable 'is_fraud' not found in test data")
        return None

    # Split features and target
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    # Print metrics
    print("Test Set Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

    # Plot ROC curve
    plot_roc_curve(y_test, y_prob)

    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_test, y_prob)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    return metrics, y_pred, y_prob


def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.MODELS_DIR, 'roc_curve.png'))
    plt.close()


def plot_precision_recall_curve(y_true, y_prob):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.MODELS_DIR, 'precision_recall_curve.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(config.MODELS_DIR, 'confusion_matrix.png'))
    plt.close()


def save_evaluation_results(metrics):
    """
    Save evaluation results to a file
    """
    results_path = os.path.join(config.MODELS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation results saved to {results_path}")


def main():
    """
    Main function to evaluate the model
    """
    # Load the model
    print("Loading the model...")
    model = load_model()
    if model is None:
        return

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    if test_data is None:
        return

    # Evaluate the model
    print("Evaluating the model...")
    metrics, _, _ = evaluate_model(model, test_data)

    # Save evaluation results
    save_evaluation_results(metrics)

    print("Model evaluation completed!")


if __name__ == "__main__":
    main()
