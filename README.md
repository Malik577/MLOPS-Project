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

### Serve the Best Model (Flask API)
```bash
python serve_model.py --model-path ../models/random_forest_best_model.pkl --port 1234
```
- Web UI: http://localhost:1234
- Health: http://localhost:1234/health
- Predict: http://localhost:1234/predict

### MLflow UI
Track experiments, compare models, and manage the model registry:
```bash
mlflow ui --port 8081
```
- Open http://localhost:8081 in your browser.

### Monitor Model Performance
```bash
python monitor_model.py --baseline
python monitor_model.py --simulate-drift feature --drift-magnitude 0.2
```

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