from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("house_rent_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("house_rent_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Ordinal encoding mappings (from your training code)
AREA_TYPE_MAPPING = {'Built Area': 1, 'Super Area': 2}
CITY_MAPPING = {
    'Mumbai': 0, 'Bangalore': 1, 'Chennai': 2,
    'Delhi': 3, 'Hyderabad': 4, 'Kolkata': 5, 'Other': 6
}
FURNISH_MAPPING = {'Semi-Furnished': 1, 'Furnished': 2}

# UPDATED: Expected columns (match exactly what training produced)
EXPECTED_COLUMNS = [
    'BHK', 'Size', 'Bathroom', 'Floor_Level', 'Total Rooms',
    'City_1.0', 'City_2.0', 'City_3.0', 'City_4.0', 'City_5.0',  # Removed City_6.0
    'Area Type_1.0', 'Area Type_2.0',
    'Furnishing Status_1.0', 'Furnishing Status_2.0'
]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Map categorical values to encoded integers
        area_type_encoded = AREA_TYPE_MAPPING[request.form["Area_Type"]]
        city_encoded = CITY_MAPPING[request.form["City"]]
        furnish_encoded = FURNISH_MAPPING[request.form["Furnishing_Status"]]

        # Create input DataFrame
        input_data = {
            'BHK': int(request.form["BHK"]),
            'Size': float(request.form["Size"]),
            'Area Type': area_type_encoded,
            'City': city_encoded,
            'Furnishing Status': furnish_encoded,
            'Bathroom': int(request.form["Bathroom"]),
            'Floor_Level': int(request.form["Floor_Level"]),
            'Total Rooms': int(request.form["BHK"]) + int(request.form["Bathroom"])
        }
        
        input_df = pd.DataFrame([input_data])

        # Apply one-hot encoding for the categorical columns with drop_first=True
        categorical_cols = ["City", "Area Type", "Furnishing Status"]
        input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Ensure all expected columns are present
        for col in EXPECTED_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder the DataFrame columns to match expected order
        input_df = input_df[EXPECTED_COLUMNS]

        # Scale features using the pre-fitted scaler
        features_scaled = scaler.transform(input_df)

        # Make prediction (model was trained on log-rent, so invert the transformation)
        log_rent_pred = model.predict(features_scaled)
        rent_pred = np.expm1(log_rent_pred)

        return render_template("index.html", 
                               prediction_text=f"Predicted Monthly Rent: ₹{rent_pred[0]:,.0f}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
