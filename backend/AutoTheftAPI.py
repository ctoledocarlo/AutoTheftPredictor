from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
import numpy as np
from sklearn import preprocessing
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Home route
@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

# Risk prediction route
@app.route('/predictrisk', methods=['POST'])
def predict_risk():
    try:
        # Load the input data from the POST request
        input_data = request.get_json()

        # Ensure the input_data is a list of dictionaries
        if not isinstance(input_data, list):
            return jsonify({"error": "Input should be a list of feature dictionaries."}), 400

        # Convert the input data to a Pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Check for required features
        required_features = ['REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'LONG_WGS84', 'LAT_WGS84', 'REPORT_HOUR_sin', 'REPORT_HOUR_cos']
        if not all(feature in input_df.columns for feature in required_features):
            missing = list(set(required_features) - set(input_df.columns))
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        # Load the trained classifier model
        classifier_model = joblib.load("random_forest_classifier.pkl")

        print("Model loaded successfully.")

        # Make predictions using the model
        predictions = classifier_model.predict(input_df[required_features])

        # Add predictions to the response
        response = {
            "predictions": predictions.tolist(),
            "details": "1 indicates high risk, 0 indicates low risk"
        }
        return jsonify(response)

    except Exception as e:
        # Handle exceptions and return a traceback for debugging
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    try:
        # Start the Flask server
        app.run(host='0.0.0.0', port=5000, debug=False)
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure 'random_forest_classifier.pkl' is in the correct directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
