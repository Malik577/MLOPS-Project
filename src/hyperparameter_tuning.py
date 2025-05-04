import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import json
import argparse
from train import load_preprocessed_data, evaluate_model

def tune_hyperparameters(model_type, param_grid, search_type='grid', n_iter=10):
    """
    Tune hyperparameters for a specified model type using grid search or random search
    
    Parameters:
    - model_type: str, type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
    - param_grid: dict, parameter grid to search
    - search_type: str, 'grid' or 'random'
    - n_iter: int, number of iterations for random search
    
    Returns:
    - best_params: dict, best parameters found
    - best_score: float, best score achieved
    """
    # Load preprocessed data
    data = load_preprocessed_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    preprocessor = data['preprocessor']
    
    # Create model class based on model_type
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model_class = RandomForestClassifier
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model_class = GradientBoostingClassifier
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model_class = LogisticRegression
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class())
    ])
    
    # Add model_ prefix to parameter names for pipeline
    pipeline_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
    
    # Create search strategy
    if search_type == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid=pipeline_param_grid,
            cv=3,  # Reduced for quicker execution
            scoring='roc_auc',
            verbose=1,
            n_jobs=-1
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=pipeline_param_grid,
            n_iter=n_iter,
            cv=3,  # Reduced for quicker execution
            scoring='roc_auc',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported search type: {search_type}")
    
    # Log experiment with MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("search_type", search_type)
        mlflow.log_param("param_grid", str(param_grid))
        
        if search_type == 'random':
            mlflow.log_param("n_iter", n_iter)
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Get best parameters and scores
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Extract model parameters (remove 'model__' prefix)
        model_best_params = {k.replace('model__', ''): v for k, v in best_params.items()}
        
        # Log best parameters and score
        for param_name, param_value in model_best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        mlflow.log_metric("best_cv_score", best_score)
        
        # Train model with best parameters
        best_model = search.best_estimator_
        
        # Evaluate on test set
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Log test metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save best model locally
        model_path = f'../models/{model_type}_best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"Hyperparameter tuning complete. Best model saved to {model_path}")
        print(f"Best parameters: {model_best_params}")
        print(f"Best CV score: {best_score:.4f}")
        print("Test metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        return model_best_params, best_score, metrics

# Parameter grids for different models - smaller for quicker demonstration
PARAM_GRIDS = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    },
    'gradient_boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'logistic_regression': {
        'C': [0.1, 1.0],
        'solver': ['lbfgs', 'liblinear']
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune hyperparameters for a specified model')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                      help='Type of model to tune')
    parser.add_argument('--search_type', type=str, default='grid',
                      choices=['grid', 'random'],
                      help='Type of search to perform (grid or random)')
    parser.add_argument('--n_iter', type=int, default=5,  # Reduced for quicker execution
                      help='Number of iterations for random search')
    parser.add_argument('--param_grid', type=str, default=None,
                      help='JSON string of parameter grid (optional, uses default if not provided)')
    
    args = parser.parse_args()
    
    # Use provided param_grid or default
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    else:
        param_grid = PARAM_GRIDS[args.model_type]
    
    tune_hyperparameters(
        args.model_type,
        param_grid,
        args.search_type,
        args.n_iter
    ) 