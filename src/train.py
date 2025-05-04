import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import argparse

def load_preprocessed_data():
    """
    Load the preprocessed data
    """
    with open('../data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def train_model(model_type, params, X_train, y_train, preprocessor):
    """
    Train a model with given parameters
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline

def run_train(model_type, params):
    """
    Main training function with MLflow tracking
    """
    # Start MLflow run
    with mlflow.start_run(run_name=model_type):
        # Load preprocessed data
        data = load_preprocessed_data()
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        preprocessor = data['preprocessor']
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train model
        model = train_model(model_type, params, X_train, y_train, preprocessor)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log parameters (including defaults)
        for param_name in model.get_params():
            mlflow.log_param(param_name, model.get_params()[param_name])
        
        # Add example tags
        mlflow.set_tag("purpose", "model_training")
        mlflow.set_tag("author", "malikalghossein")
        mlflow.set_tag("model_type", model_type)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model and register in Model Registry
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_type)
        
        # Save model locally
        model_path = f'../models/{model_type}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log an extra artifact (model_info.txt)
        info_path = f'../models/{model_type}_model_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Model type: {model_type}\n")
            f.write(f"Parameters: {model.get_params()}\n")
            f.write(f"Metrics: {metrics}\n")
        mlflow.log_artifact(info_path)
        
        print(f"Model training complete. Model saved to {model_path}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified parameters')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                        help='Type of model to train')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON string of model parameters')
    
    args = parser.parse_args()
    import json
    params = json.loads(args.params)
    
    run_train(args.model_type, params) 