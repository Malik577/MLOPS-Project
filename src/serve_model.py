#!/usr/bin/env python3

import os
import pickle
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Global variables for the model
model = None
model_path = None

# HTML template for the home page
HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bank Marketing Model Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
        }
        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        label {
            font-weight: bold;
        }
        input, select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            grid-column: span 2;
            margin-top: 15px;
        }
        button:hover {
            background-color: #3498db;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4f8;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bank Marketing Model Prediction</h1>
        
        <form id="predictionForm" onsubmit="submitForm(event)">
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" value="35" required>
            
            <label for="job">Job:</label>
            <select id="job" name="job" required>
                <option value="admin.">Admin</option>
                <option value="blue-collar">Blue Collar</option>
                <option value="entrepreneur">Entrepreneur</option>
                <option value="housemaid">Housemaid</option>
                <option value="management" selected>Management</option>
                <option value="retired">Retired</option>
                <option value="self-employed">Self-employed</option>
                <option value="services">Services</option>
                <option value="student">Student</option>
                <option value="technician">Technician</option>
                <option value="unemployed">Unemployed</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="marital">Marital Status:</label>
            <select id="marital" name="marital" required>
                <option value="divorced">Divorced</option>
                <option value="married" selected>Married</option>
                <option value="single">Single</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="education">Education:</label>
            <select id="education" name="education" required>
                <option value="basic.4y">Basic 4y</option>
                <option value="basic.6y">Basic 6y</option>
                <option value="basic.9y">Basic 9y</option>
                <option value="high.school">High School</option>
                <option value="illiterate">Illiterate</option>
                <option value="professional.course">Professional Course</option>
                <option value="university.degree" selected>University Degree</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="default">Has Credit Default?</label>
            <select id="default" name="default" required>
                <option value="no" selected>No</option>
                <option value="yes">Yes</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="balance">Account Balance:</label>
            <input type="number" id="balance" name="balance" value="5000" required>
            
            <label for="housing">Has Housing Loan?</label>
            <select id="housing" name="housing" required>
                <option value="no">No</option>
                <option value="yes" selected>Yes</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="loan">Has Personal Loan?</label>
            <select id="loan" name="loan" required>
                <option value="no" selected>No</option>
                <option value="yes">Yes</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="contact">Contact Method:</label>
            <select id="contact" name="contact" required>
                <option value="cellular" selected>Cellular</option>
                <option value="telephone">Telephone</option>
                <option value="unknown">Unknown</option>
            </select>
            
            <label for="day">Day of Month:</label>
            <input type="number" id="day" name="day" min="1" max="31" value="15" required>
            
            <label for="month">Month:</label>
            <select id="month" name="month" required>
                <option value="jan">January</option>
                <option value="feb">February</option>
                <option value="mar">March</option>
                <option value="apr">April</option>
                <option value="may" selected>May</option>
                <option value="jun">June</option>
                <option value="jul">July</option>
                <option value="aug">August</option>
                <option value="sep">September</option>
                <option value="oct">October</option>
                <option value="nov">November</option>
                <option value="dec">December</option>
            </select>
            
            <label for="duration">Last Contact Duration (seconds):</label>
            <input type="number" id="duration" name="duration" value="300" required>
            
            <label for="campaign">Campaign Contacts:</label>
            <input type="number" id="campaign" name="campaign" value="2" required>
            
            <label for="pdays">Days Since Last Contact:</label>
            <input type="number" id="pdays" name="pdays" value="-1" required>
            
            <label for="previous">Previous Contacts:</label>
            <input type="number" id="previous" name="previous" value="0" required>
            
            <label for="poutcome">Previous Outcome:</label>
            <select id="poutcome" name="poutcome" required>
                <option value="failure">Failure</option>
                <option value="success">Success</option>
                <option value="unknown" selected>Unknown</option>
            </select>
            
            <button type="submit">Predict</button>
        </form>
        
        <div id="result" class="result" style="display:none;">
            <h2>Prediction Result</h2>
            <p id="predictionResult"></p>
            <p id="probabilityResult"></p>
        </div>
    </div>

    <script>
        function submitForm(event) {
            event.preventDefault();
            
            const form = document.getElementById('predictionForm');
            const formData = new FormData(form);
            const data = {};
            
            // Extract form values
            for (let [key, value] of formData.entries()) {
                data[key] = value;
            }
            
            // Convert numeric fields to numbers
            const numericFields = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'];
            numericFields.forEach(field => {
                data[field] = parseInt(data[field]);
            });
            
            // Create the payload in MLflow format
            const payload = {
                columns: [
                    "age", "job", "marital", "education", "default", "balance", "housing", 
                    "loan", "contact", "day", "month", "duration", "campaign", "pdays", 
                    "previous", "poutcome"
                ],
                data: [
                    [
                        data.age, data.job, data.marital, data.education, data.default, 
                        data.balance, data.housing, data.loan, data.contact, data.day, 
                        data.month, data.duration, data.campaign, data.pdays, 
                        data.previous, data.poutcome
                    ]
                ]
            };
            
            // Make API call
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            })
            .then(response => response.json())
            .then(data => {
                // Display results
                const resultDiv = document.getElementById('result');
                const predictionText = document.getElementById('predictionResult');
                const probabilityText = document.getElementById('probabilityResult');
                
                const prediction = data.predictions[0] === 1 ? 'YES' : 'NO';
                const probability = (data.probabilities[0] * 100).toFixed(2);
                
                predictionText.innerHTML = `<strong>Will subscribe to term deposit:</strong> ${prediction}`;
                probabilityText.innerHTML = `<strong>Probability of subscribing:</strong> ${probability}%`;
                
                resultDiv.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while making the prediction.');
            });
        }
    </script>
</body>
</html>
'''

def load_model(model_path):
    """
    Load a trained model from a pickle file
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint with a simple UI
    """
    return render_template_string(HOME_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    if model is not None:
        return jsonify({"status": "healthy", "model": model_path}), 200
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    """
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get input data
    if not request.is_json:
        return jsonify({"error": "Input must be JSON"}), 400
    
    data = request.get_json()
    
    try:
        # Extract data from request
        if 'columns' in data and 'data' in data:
            # MLflow-style input format
            df = pd.DataFrame(data['data'], columns=data['columns'])
        elif isinstance(data, dict) and not ('columns' in data or 'data' in data):
            # Single record as a flat JSON object
            df = pd.DataFrame([data])
        else:
            return jsonify({"error": "Invalid input format"}), 400
        
        # Make prediction
        probabilities = model.predict_proba(df)[:, 1]
        predictions = model.predict(df)
        
        # Format response
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

def main(model_path, port=1234):
    """
    Start the Flask app with the specified model
    """
    global model
    model = load_model(model_path)
    
    if model is None:
        logger.error(f"Could not load model from {model_path}")
        return
    
    # Print usage examples
    print("\nModel serving started!")
    print(f"API is running at http://localhost:{port}")
    print("\nExample curl commands:")
    print(f"Health check: curl http://localhost:{port}/health")
    
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
    
    print(f"Prediction (MLflow format): curl -X POST -H \"Content-Type:application/json\" --data '{json.dumps(sample_json)}' http://localhost:{port}/predict")
    print(f"Web interface: http://localhost:{port}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serve a machine learning model using Flask")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pickled model file")
    parser.add_argument("--port", type=int, default=1234, help="Port to run the server on")
    
    args = parser.parse_args()
    model_path = args.model_path
    
    main(args.model_path, args.port) 