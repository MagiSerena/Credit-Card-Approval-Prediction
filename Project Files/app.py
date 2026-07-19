from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
import os
import json
from io import StringIO
import traceback

app = Flask(__name__)

# Constants for validation
FEATURES_CONFIG = {
    "Age": {"type": "float", "min": 18, "max": 100},
    "Income": {"type": "float", "min": 0, "max": 1000000},
    "Employment": {"type": "categorical", "allowed": ["Yes", "No"]},
    "CreditScore": {"type": "float", "min": 300, "max": 850},
    "LoanAmount": {"type": "float", "min": 500, "max": 1000000},
    "Married": {"type": "categorical", "allowed": ["Yes", "No"]}
}

MODEL_LOADED = False
model = None
scaler = None
le_emp = None
le_mar = None

def load_models():
    global model, scaler, le_emp, le_mar, MODEL_LOADED
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_emp = joblib.load('models/le_emp.pkl')
        le_mar = joblib.load('models/le_mar.pkl')
        MODEL_LOADED = True
    except Exception as e:
        print("Warning: Models not found. Please run train_model.py first.")
        MODEL_LOADED = False

load_models()

def validate_input(data):
    """ Validate input data based on FEATURES_CONFIG """
    errors = []
    validated_data = {}
    
    for feature, rules in FEATURES_CONFIG.items():
        if feature not in data:
            errors.append(f"Missing required feature: '{feature}'")
            continue
            
        val = data[feature]
        
        if pd.isna(val) or val == "":
            errors.append(f"Field '{feature}' cannot be empty")
            continue
            
        if rules["type"] == "float":
            try:
                numeric_val = float(val)
                if numeric_val < rules["min"] or numeric_val > rules["max"]:
                    errors.append(f"Feature '{feature}' value {numeric_val} is out of bounds ({rules['min']}-{rules['max']})")
                else:
                    validated_data[feature] = numeric_val
            except ValueError:
                errors.append(f"Feature '{feature}' must be a numeric value")
                
        elif rules["type"] == "categorical":
            str_val = str(val).strip()
            if str_val not in rules["allowed"]:
                errors.append(f"Feature '{feature}' value '{str_val}' invalid. Allowed: {rules['allowed']}")
            else:
                validated_data[feature] = str_val
                
    return validated_data, errors

def make_prediction(features_dict):
    """ Helper to run prediction on validated dict """
    # Encode categorical
    emp_encoded = le_emp.transform([features_dict['Employment']])[0]
    mar_encoded = le_mar.transform([features_dict['Married']])[0]
    
    # Create DataFrame
    df_pred = pd.DataFrame({
        'Age': [features_dict['Age']],
        'Income': [features_dict['Income']],
        'Employment': [emp_encoded],
        'CreditScore': [features_dict['CreditScore']],
        'LoanAmount': [features_dict['LoanAmount']],
        'Married': [mar_encoded]
    })
    
    # Scale features
    features_scaled = scaler.transform(df_pred)
    
    # Predict
    prediction = int(model.predict(features_scaled)[0])
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        prob_array = model.predict_proba(features_scaled)[0]
        # prob_array[0] is prob of class 0 (Approved), prob_array[1] is prob of class 1 (Rejected)
        # return highest prob
        probability = float(prob_array[prediction])
        
    return prediction, probability

# ----------------- REST APIs ----------------- #

@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if MODEL_LOADED else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": MODEL_LOADED
    }), 200

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        "features": list(FEATURES_CONFIG.keys()),
        "constraints": FEATURES_CONFIG
    }), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not ready. Please train model."}), 503
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        validated_data, errors = validate_input(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400
            
        prediction, probability = make_prediction(validated_data)
        
        status = "Approved" if prediction == 0 else "Rejected"
        confidence = f"{probability*100:.2f}%" if probability else "N/A"
        
        return jsonify({
            "prediction": prediction,
            "status": status,
            "confidence": confidence,
            "probability": probability
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e), "trace": traceback.format_exc()}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not ready."}), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No CSV file uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
        
    try:
        df = pd.read_csv(file)
        # Check required columns
        missing_cols = [col for col in FEATURES_CONFIG.keys() if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400
            
        results = []
        for index, row in df.iterrows():
            data_dict = row.to_dict()
            validated_data, errors = validate_input(data_dict)
            if errors:
                results.append({"row": index, "status": "Failed", "errors": errors})
            else:
                pred, prob = make_prediction(validated_data)
                results.append({
                    "row": index, 
                    "status": "Approved" if pred == 0 else "Rejected", 
                    "confidence": prob
                })
                
        return jsonify({"batch_results": results}), 200
    except Exception as e:
        return jsonify({"error": "Failed to process batch", "detail": str(e)}), 500


# ----------------- UI Routes ----------------- #

@app.route('/', methods=['GET'])
def index():
    # Gather model metrics if available
    metrics = None
    if os.path.exists('models/model_metrics.csv'):
        metrics_df = pd.read_csv('models/model_metrics.csv')
        metrics = metrics_df.to_dict(orient='records')
        
    # Gather images
    eda_images = []
    if os.path.exists('static/images'):
        for file in os.listdir('static/images'):
            if file.endswith('.png'):
                eda_images.append(file)
                
    # Gather user data
    user_data = []
    if os.path.exists('application_record.csv'):
        users_df = pd.read_csv('application_record.csv')
        user_data = users_df.to_dict(orient='records')
            
    return render_template('index.html', eda_images=eda_images, metrics=metrics, users=user_data)

if __name__ == '__main__':
    app.run(debug=True)
