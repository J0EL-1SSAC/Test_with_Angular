from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
from datetime import datetime
import pytz  # Required to handle timezones correctly

# Initialize Flask App
app = Flask(__name__)

# --- MODEL AND DATA LOADING ---

# Load the trained machine learning model
try:
    with open('crowd_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'crowd_model.pkl' not found. Please run train_model.py first.")
    model = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    model = None

# List of all stations (must match the stations the model was trained on)
stations = sorted(list(set([
    "Wimco Nagar Depot", "Wimco Nagar", "Tiruvottiyur", "Tiruvottiyur Theradi", "Kaladipet",
    "Tollgate", "Tondiarpet", "Sir Theagaraya College", "Washermanpet", "Mannadi",
    "High Court", "Puratchi Thalaivar Dr. M.G. Ramachandran Central", "Government Estate",
    "LIC", "Thousand Lights", "AG-DMS", "Teynampet", "Nandanam", "Saidapet", "Little Mount",
    "Guindy", "Alandur", "Nanganallur Road", "Meenambakkam", "Chennai International Airport",
    "St. Thomas Mount", "Ashok Nagar", "Ekkattuthangal", "Arumbakkam",
    "Vadapalani", "Koyambedu", "Thirumangalam", "Anna Nagar Tower",
    "Anna Nagar East", "Shenoy Nagar", "Pachaiyappa's College", "Kilpauk", "Nehru Park",
    "Egmore", "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT"
])))


# --- FLASK ROUTES ---

# Main route to render the website's homepage
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

# API route to get the list of stations for the filter dropdown
@app.route('/api/stations')
def get_stations():
    """Returns a sorted list of all metro stations."""
    return jsonify(stations)

# API route to get real-time crowd predictions
@app.route('/api/crowd-data')
def get_crowd_data():
    """
    This is the core API endpoint. It generates predictions for the current time
    and returns them as JSON, allowing the frontend to update without reloading.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded. Cannot make predictions."}), 500

    # Get the current time for Chennai (India Standard Time)
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)

    # Extract features from the current time
    hour = now.hour
    day_of_week = now.weekday()  # Monday=0, Sunday=6
    is_weekend = 1 if day_of_week >= 5 else 0
    # NOTE: A robust solution for holidays would use a library like 'holidays'
    # For this hackathon project, we'll assume it's not a holiday for live predictions.
    is_holiday = 0
    is_peak_hour = 1 if (7 <= hour < 11) or (16 <= hour < 20) else 0

    # Create a DataFrame with the current time's features for ALL stations
    # This is much more efficient than predicting one by one in a loop
    data_to_predict = {
        'station': stations,
        'hour': [hour] * len(stations),
        'day_of_week': [day_of_week] * len(stations),
        'is_weekend': [is_weekend] * len(stations),
        'is_holiday': [is_holiday] * len(stations),
        'is_peak_hour': [is_peak_hour] * len(stations),
    }
    df_predict = pd.DataFrame(data_to_predict)

    # Use the loaded model to make predictions
    try:
        crowd_levels = model.predict(df_predict)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Format the predictions into a user-friendly JSON structure
    predictions = [
        {
            "station": station,
            # Clamp the prediction between 0 and 1, as models can sometimes predict slightly outside this range
            "crowd_level": max(0.0, min(1.0, crowd_level))
        }
        for station, crowd_level in zip(stations, crowd_levels)
    ]

    return jsonify(predictions)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Runs the Flask web server
    app.run(debug=True)