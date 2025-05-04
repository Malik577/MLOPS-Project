# Bank Marketing MLOps Project

Predict whether a client will subscribe to a term deposit using a robust, production-ready MLOps pipeline powered by MLflow.

---

## Project Structure

```
Mlops1-project/
├── data/                  # Processed and raw data
├── bank-additional/       # Raw data files from UCI
├── models/                # Best and latest model files
├── monitoring/            # Monitoring outputs and drift plots
├── src/                   # All source code (training, serving, monitoring)
│   ├── tests/            # Unit tests
│   ├── train.py          # Model training script
│   ├── serve_model.py    # Flask API server
│   └── monitor_model.py  # Monitoring script
├── mlruns/                # MLflow experiment tracking (auto-generated)
├── .venv/                 # Python virtual environment
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Mlops1-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Download the dataset:
   The project expects data in `bank-additional/bank-additional-full.csv`.
   [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## How to Run

### Train Models
Train and log a model (e.g., Random Forest):
```bash
cd src
python train.py --model_type random_forest --params '{}'
```

### Hyperparameter Tuning
Find the best model parameters:
```bash
python hyperparameter_tuning.py --model_type random_forest --search_type random --n_iter 20
```

### MLflow UI and Model Registry

1. **Start MLflow UI**:
   ```bash
   mlflow ui --port 8081
   ```
   Access the UI at http://localhost:8081

2. **Model Registry Usage**:
   - **View Models**: Navigate to the "Models" tab in MLflow UI
   - **Promote Models**:
     ```python
     # Using MLflow Python API
     import mlflow
     
     # Promote model to Staging
     mlflow.register_model(
         "runs:/<run_id>/model",
         "BankMarketingModel",
         tags={"stage": "Staging"}
     )
     
     # Promote to Production
     client = mlflow.tracking.MlflowClient()
     client.transition_model_version_stage(
         name="BankMarketingModel",
         version=1,
         stage="Production"
     )
     ```

3. **Model Stages**:
   - **None**: Initial state
   - **Staging**: Testing environment
   - **Production**: Live environment
   - **Archived**: Retired models

### Model Serving & API Endpoints

Our Flask API provides a user-friendly interface for model predictions and monitoring. Here's how to use it:

1. Start the server:
```bash
python serve_model.py --model-path ../models/random_forest_best_model.pkl --port 1234
```

2. Available Endpoints:

   - **Web Interface**: http://localhost:1234
     - User-friendly form to input prediction data
     - Real-time prediction results
     - Model performance metrics display

   - **Health Check**: http://localhost:1234/health
     - Returns server status and model loading status
     - Example response:
     ```json
     {
       "status": "healthy",
       "model_loaded": true,
       "timestamp": "2024-03-21T10:30:00Z"
     }
     ```

   - **Prediction API**: http://localhost:1234/predict
     - Method: POST
     - Content-Type: application/json
     - Example request:
     ```json
     {
       "age": 41,
       "job": "management",
       "marital": "married",
       "education": "tertiary",
       "default": "no",
       "balance": 1000,
       "housing": "yes",
       "loan": "no",
       "contact": "cellular",
       "day": 5,
       "month": "may",
       "duration": 300,
       "campaign": 1,
       "pdays": -1,
       "previous": 0,
       "poutcome": "unknown"
     }
     ```
     - Example response:
     ```json
     {
       "prediction": 1,
       "probability": 0.85,
       "timestamp": "2024-03-21T10:30:00Z"
     }
     ```

### Model Monitoring

Our monitoring system tracks model performance and data drift in real-time. Here's how to use it:

1. **Baseline Performance**:
```bash
python monitor_model.py --baseline
```
This creates a baseline of model performance metrics and feature distributions.

2. **Simulate Drift**:
```bash
# Simulate feature drift
python monitor_model.py --simulate-drift feature --drift-magnitude 0.2

# Simulate label drift
python monitor_model.py --simulate-drift label --drift-magnitude 0.3
```

3. **Monitor Real-time Performance**:
```bash
python monitor_model.py --real-time --interval 3600
```
This runs continuous monitoring with hourly checks.

4. **Monitoring Outputs**:
   - Drift detection plots in `monitoring/drift_plots/`
   - Performance metrics in `monitoring/metrics/`
   - Alert logs in `monitoring/alerts/`

5. **Key Metrics Tracked**:
   - Prediction accuracy
   - Feature drift scores
   - Label distribution changes
   - Model confidence scores
   - API response times

### Testing

Run the test suite to ensure code reliability:

```bash
# Run all tests
python -m pytest src/tests/

# Run specific test file
python -m pytest src/tests/test_model.py

# Run with coverage report
python -m pytest --cov=src src/tests/
```

Key test areas:
- Model training and prediction
- API endpoints
- Data preprocessing
- Monitoring functionality
- Model registry operations

---

## Results

- Best Model: Random Forest (after tuning)
- Accuracy: ~0.90
- All experiments, metrics, and model versions are tracked in MLflow.
- Drift detection and monitoring visualized in the `monitoring/` folder.

---

## Features

- End-to-end ML pipeline: Data prep, training, tuning, serving, monitoring.
- MLflow integration: Experiment tracking, model registry, UI.
- Production-ready Flask API: For real-time predictions.
- Model monitoring: Detects drift and logs alerts/plots.
- Clean, human-friendly structure: Easy to navigate and extend.
- Comprehensive test suite: Ensures code reliability.

---

## Dataset

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- 45,211 instances, 16 features, binary classification (term deposit subscription)

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue or submit a PR.

---

## License

This project is for educational purposes. 