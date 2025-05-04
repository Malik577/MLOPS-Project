#!/usr/bin/env python3

import os
import mlflow
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from compare_models import load_model, load_test_data

def evaluate_models():
    """
    Evaluate all models and return the best one based on ROC AUC
    """
    model_paths = [
        '../models/random_forest_model.pkl',
        '../models/random_forest_best_model.pkl',
        '../models/logistic_regression_model.pkl',
        '../models/logistic_regression_best_model.pkl',
        '../models/gradient_boosting_model.pkl',
        '../models/gradient_boosting_best_model.pkl'
    ]
    
    display_names = [
        'Random Forest (Baseline)',
        'Random Forest (Tuned)',
        'Logistic Regression (Baseline)',
        'Logistic Regression (Tuned)',
        'Gradient Boosting (Baseline)',
        'Gradient Boosting (Tuned)'
    ]
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Evaluate all models
    results = []
    for model_path, display_name in zip(model_paths, display_names):
        try:
            model = load_model(model_path)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            results.append({
                'model_path': model_path,
                'display_name': display_name,
                'roc_auc': roc_auc,
                'accuracy': accuracy
            })
            
            print(f"Model: {display_name}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error evaluating model {display_name}: {e}")
    
    # Find the best model based on ROC AUC
    best_model_info = max(results, key=lambda x: x['roc_auc'])
    print(f"\nBest model: {best_model_info['display_name']}")
    print(f"  ROC AUC: {best_model_info['roc_auc']:.4f}")
    print(f"  Accuracy: {best_model_info['accuracy']:.4f}")
    
    return best_model_info


def package_model(model_path, model_name):
    """
    Package the model using MLflow
    """
    model = load_model(model_path)
    signature = create_model_signature()
    
    # Save the model using MLflow's model registry
    with mlflow.start_run(run_name=f"deploy_{model_name}"):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature
        )
        
        print(f"Model '{model_name}' registered with MLflow")
        return model


def deploy_model(model_path, model_name, port=1234):
    """
    Deploy the model as a REST service using MLflow
    """
    # Package the model
    model = package_model(model_path, model_name)
    
    # Create a sample input for the model
    sample_input = create_sample_input()
    print("\nSample input for prediction:")
    print(sample_input)
    
    # Make prediction with the loaded model
    prediction = model.predict_proba(sample_input)[0][1]
    print(f"Sample prediction (probability): {prediction:.4f}")
    
    # Create deployment command
    deploy_cmd = f"mlflow models serve -m 'models:/{model_name}/latest' -p {port}"
    
    print("\nDeployment complete!")
    print(f"To start the prediction service, run:")
    print(f"  {deploy_cmd}")
    print(f"\nThen you can make predictions using HTTP requests:")
    print(f"  curl -X POST -H \"Content-Type:application/json\" --data '{create_sample_json()}' http://127.0.0.1:{port}/invocations")
    
    return deploy_cmd


def create_model_signature():
    """
    Create an MLflow model signature for the input and output schema
    """
    from mlflow.models.signature import infer_signature
    
    # Load a sample input
    sample_input = create_sample_input()
    
    # Load any model to generate prediction samples
    X_test, _ = load_test_data()
    model = load_model('../models/random_forest_model.pkl')
    sample_output = model.predict(sample_input)
    
    # Create signature
    signature = infer_signature(sample_input, sample_output)
    return signature


def create_sample_input():
    """
    Create a sample input for the model
    """
    sample_data = {
        'age': [35],
        'job': ['management'],
        'marital': ['married'],
        'education': ['university.degree'],
        'default': ['no'],
        'balance': [5000],
        'housing': ['yes'],
        'loan': ['no'],
        'contact': ['cellular'],
        'day': [15],
        'month': ['may'],
        'duration': [300],
        'campaign': [2],
        'pdays': [-1],
        'previous': [0],
        'poutcome': ['unknown']
    }
    
    return pd.DataFrame(sample_data)


def create_sample_json():
    """
    Create a sample JSON input for the deployed model
    """
    sample_json = {
        "columns": [
            "age", "job", "marital", "education", "default", "balance", "housing", 
            "loan", "contact", "day", "month", "duration", "campaign", "pdays", 
            "previous", "poutcome"
        ],
        "data": [
            [
                35, "management", "married", "university.degree", "no", 5000, "yes", 
                "no", "cellular", 15, "may", 300, 2, -1, 0, "unknown"
            ]
        ]
    }
    
    import json
    return json.dumps(sample_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy a model as a REST service using MLflow')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate all models to find the best one')
    parser.add_argument('--model-path', type=str, help='Path to the model to deploy')
    parser.add_argument('--model-name', type=str, default='bank_marketing_model', help='Name for the deployed model')
    parser.add_argument('--port', type=int, default=1234, help='Port for the prediction service')
    
    args = parser.parse_args()
    
    if args.evaluate:
        best_model_info = evaluate_models()
        model_path = best_model_info['model_path']
        model_name = args.model_name
    elif args.model_path:
        model_path = args.model_path
        model_name = args.model_name
    else:
        print("Either --evaluate or --model-path must be specified")
        exit(1)
    
    deploy_model(model_path, model_name, args.port) 