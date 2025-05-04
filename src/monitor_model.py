#!/usr/bin/env python3

import os
import sys
import pickle
import pandas as pd
import numpy as np
import time
import datetime
import json
import mlflow
from mlflow.tracking import MlflowClient
import argparse
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = os.path.join("..", "bank-additional", "bank-additional-full.csv")
MODELS_DIR = os.path.join("..", "models")
MONITORING_DIR = os.path.join("..", "monitoring")
MONITORING_MODEL_PATH = os.path.join(MONITORING_DIR, "monitoring_model.pkl")


def load_data(test_size=0.2, random_state=42):
    """Load and split the data for testing"""
    logger.info(f"Loading data from {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, delimiter=';')
    
    # Convert target to binary
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    
    # Split data for monitoring
    X = data.drop('y', axis=1)
    y = data['y']
    
    # For simulation, we'll use the test set as new data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def preprocess_data(data):
    """
    Basic preprocessing for monitoring
    """
    # Rename columns to match convention if needed
    if 'day_of_week' in data.columns and 'day' not in data.columns:
        data = data.rename(columns={'day_of_week': 'day'})
    
    # Handle categorical variables
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                        'contact', 'month', 'day', 'poutcome']
    
    # For numerical features, fill NaN values
    numerical_cols = data.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # For categorical features, one-hot encode
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
    
    return data_encoded


def train_monitoring_model(X_train, y_train):
    """Train a simple model for monitoring purposes"""
    logger.info("Training monitoring model")
    
    # Preprocess training data
    X_train_processed = preprocess_data(X_train)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MONITORING_MODEL_PATH), exist_ok=True)
    with open(MONITORING_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the column information
    columns_file = os.path.join(MONITORING_DIR, "monitoring_columns.json")
    with open(columns_file, 'w') as f:
        json.dump(X_train_processed.columns.tolist(), f)
    
    logger.info(f"Monitoring model saved to {MONITORING_MODEL_PATH}")
    logger.info(f"Column information saved to {columns_file}")
    
    return model


def load_monitoring_model():
    """Load the monitoring model"""
    if not os.path.exists(MONITORING_MODEL_PATH):
        logger.error(f"Monitoring model not found at {MONITORING_MODEL_PATH}")
        return None
    
    try:
        with open(MONITORING_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Monitoring model loaded from {MONITORING_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading monitoring model: {e}")
        return None


def prepare_monitoring_data(X):
    """Prepare data for monitoring to match the trained model's expectations"""
    X_processed = preprocess_data(X)
    
    # Load expected columns
    columns_file = os.path.join(MONITORING_DIR, "monitoring_columns.json")
    if os.path.exists(columns_file):
        with open(columns_file, 'r') as f:
            expected_columns = json.load(f)
        
        # Align columns with expected
        missing_cols = set(expected_columns) - set(X_processed.columns)
        extra_cols = set(X_processed.columns) - set(expected_columns)
        
        # Add missing columns
        for col in missing_cols:
            X_processed[col] = 0
        
        # Remove extra columns
        for col in extra_cols:
            X_processed = X_processed.drop(col, axis=1)
        
        # Reorder columns
        X_processed = X_processed[expected_columns]
    
    return X_processed


def evaluate_model(model, X, y):
    """
    Evaluate the model and return metrics
    """
    # Make predictions
    X_processed = prepare_monitoring_data(X)
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def simulate_drift(X, y, drift_type='feature', magnitude=0.1):
    """
    Simulate drift in the data for demonstration purposes
    
    Parameters:
    drift_type: 'feature' or 'label' 
    magnitude: strength of the drift (0.0 to 1.0)
    """
    X_drift = X.copy()
    y_drift = y.copy()
    
    if drift_type == 'feature':
        # Simulate feature drift by shifting numerical features
        numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        for col in numerical_cols:
            if col in X_drift.columns:
                # Add a shift based on the magnitude
                mean_val = X_drift[col].mean()
                std_val = X_drift[col].std()
                X_drift[col] = X_drift[col] + magnitude * std_val * 2
                
        logger.info(f"Simulated feature drift with magnitude {magnitude}")
                
    elif drift_type == 'label':
        # Simulate label drift by flipping some labels
        flip_mask = np.random.random(size=len(y_drift)) < magnitude
        y_drift[flip_mask] = 1 - y_drift[flip_mask]
        logger.info(f"Simulated label drift with magnitude {magnitude}, flipped {flip_mask.sum()} labels")
    
    return X_drift, y_drift


def plot_metric_over_time(metrics_list, metric_name, output_file):
    """Create a plot of a metric over time"""
    timestamps = [pd.to_datetime(m['timestamp']) for m in metrics_list]
    metric_values = [m[metric_name] for m in metrics_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, metric_values, marker='o', linestyle='-')
    plt.title(f'{metric_name.capitalize()} Over Time')
    plt.xlabel('Time')
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    logger.info(f"Saved metric plot to {output_file}")


def detect_drift(baseline_metrics, current_metrics, threshold=0.05):
    """
    Detect if there is drift in the model performance
    
    Returns:
    - is_drift_detected: True if drift is detected
    - drift_metrics: List of metrics that show drift
    """
    if not baseline_metrics:
        logger.info("No baseline metrics to compare against")
        return False, []
    
    drift_metrics = []
    is_drift_detected = False
    
    # Check each metric for drift
    for metric in ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']:
        baseline_value = baseline_metrics[metric]
        current_value = current_metrics[metric]
        
        # Calculate absolute difference
        difference = abs(current_value - baseline_value)
        
        # Calculate percent change
        if baseline_value > 0:
            percent_change = difference / baseline_value
        else:
            percent_change = float('inf')
            
        if percent_change > threshold:
            drift_metrics.append({
                'metric': metric,
                'baseline_value': baseline_value,
                'current_value': current_value,
                'percent_change': percent_change
            })
            is_drift_detected = True
            
    return is_drift_detected, drift_metrics


def log_to_mlflow(metrics, run_name):
    """Log metrics to MLflow"""
    # Set the experiment
    mlflow.set_experiment("model_monitoring")
    
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log metrics
        for key, value in metrics.items():
            if key != 'timestamp':
                mlflow.log_metric(key, value)
        
        # Log parameters
        mlflow.log_param("timestamp", metrics['timestamp'])
        
        # Log a note about the monitoring
        mlflow.set_tag("monitoring_type", "performance")
        
        logger.info(f"Metrics logged to MLflow run: {mlflow.active_run().info.run_id}")


def run_monitoring_cycle(baseline=False, simulate=False, drift_type=None, drift_magnitude=0.1):
    """Run a single monitoring cycle"""
    
    # Create monitoring directory if it doesn't exist
    os.makedirs(MONITORING_DIR, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Load or train monitoring model
    if baseline or not os.path.exists(MONITORING_MODEL_PATH):
        logger.info("Training baseline monitoring model")
        model = train_monitoring_model(X_train, y_train)
        
        # Evaluate on training data to get baseline
        baseline_metrics = evaluate_model(model, X_train, y_train)
        
        # Save baseline metrics
        baseline_file = os.path.join(MONITORING_DIR, "baseline_metrics.json")
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
        
        logger.info(f"Baseline metrics saved to {baseline_file}")
        
        # Log baseline to MLflow
        log_to_mlflow(baseline_metrics, "baseline_metrics")
    else:
        # Load model
        model = load_monitoring_model()
        if model is None:
            logger.error("Failed to load monitoring model. Please run with --baseline to create a new one.")
            return
    
    # Load baseline metrics
    baseline_file = os.path.join(MONITORING_DIR, "baseline_metrics.json")
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)
    else:
        baseline_metrics = None
    
    # Apply drift simulation if requested
    test_data = X_test.copy()
    test_labels = y_test.copy()
    
    if simulate and drift_type:
        test_data, test_labels = simulate_drift(test_data, test_labels, drift_type, drift_magnitude)
    
    # Evaluate model on test data
    current_metrics = evaluate_model(model, test_data, test_labels)
    
    # Log monitoring metrics to MLflow
    run_name = f"monitoring_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_to_mlflow(current_metrics, run_name)
    
    # Only check for drift if we have baseline metrics
    if baseline_metrics:
        # Detect drift
        is_drift, drift_metrics = detect_drift(baseline_metrics, current_metrics)
        
        if is_drift:
            logger.warning("DRIFT DETECTED!")
            for dm in drift_metrics:
                logger.warning(f"  {dm['metric']}: {dm['baseline_value']:.4f} -> {dm['current_value']:.4f} " 
                             f"(change: {dm['percent_change']*100:.2f}%)")
            
            # Log drift alert to MLflow
            with mlflow.start_run(run_name=f"drift_alert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("timestamp", datetime.datetime.now().isoformat())
                mlflow.set_tag("alert_type", "drift_detected")
                
                for dm in drift_metrics:
                    mlflow.log_metric(f"{dm['metric']}_baseline", dm['baseline_value'])
                    mlflow.log_metric(f"{dm['metric']}_current", dm['current_value'])
                    mlflow.log_metric(f"{dm['metric']}_change", dm['percent_change'])
    
    # Load monitoring history if exists
    history_file = os.path.join(MONITORING_DIR, "metrics_history.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            metrics_history = json.load(f)
    else:
        metrics_history = []
    
    # Add current metrics to history
    metrics_history.append(current_metrics)
    
    # Save updated history to file
    with open(history_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # Create plots for key metrics if we have enough data
    if len(metrics_history) > 1:
        for metric in ['accuracy', 'roc_auc', 'f1']:
            plot_file = os.path.join(MONITORING_DIR, f"{metric}_over_time.png")
            plot_metric_over_time(metrics_history, metric, plot_file)


def main():
    parser = argparse.ArgumentParser(description="Monitor model performance over time")
    parser.add_argument("--baseline", action="store_true", 
                      help="Establish baseline metrics by training a new monitoring model")
    parser.add_argument("--simulate-drift", choices=['none', 'feature', 'label'], default='none',
                      help="Simulate drift for testing")
    parser.add_argument("--drift-magnitude", type=float, default=0.1,
                      help="Magnitude of simulated drift (0.0 to 1.0)")
    parser.add_argument("--interval", type=int, default=0,
                      help="Interval between monitoring cycles in seconds (0 for one-time run)")
    
    args = parser.parse_args()
    
    # Run monitoring cycle
    if args.interval > 0:
        logger.info(f"Starting continuous monitoring with interval of {args.interval} seconds")
        
        try:
            # First run might be baseline
            run_monitoring_cycle(
                baseline=args.baseline,
                simulate=args.simulate_drift != 'none',
                drift_type=args.simulate_drift,
                drift_magnitude=args.drift_magnitude
            )
            
            # Subsequent runs
            while True:
                logger.info(f"Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
                
                run_monitoring_cycle(
                    baseline=False,  # Never baseline for subsequent runs
                    simulate=args.simulate_drift != 'none',
                    drift_type=args.simulate_drift,
                    drift_magnitude=args.drift_magnitude
                )
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    else:
        # Run once
        run_monitoring_cycle(
            baseline=args.baseline,
            simulate=args.simulate_drift != 'none',
            drift_type=args.simulate_drift,
            drift_magnitude=args.drift_magnitude
        )
        logger.info("Monitoring cycle completed")


if __name__ == "__main__":
    main() 