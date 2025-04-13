# python -m venv myenv
# myenv\Scripts\activate
# pip3 install flask numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0 gunicorn 
# pip3 freeze > requirements.txt
# python app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# 🌋 Load Earthquake model, scaler, and feature names
with open("earthquake_classifier.pkl", "rb") as f:
    earthquake_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    eq_features = pickle.load(f)

scale_cols = ['longitude', 'latitude', 'depth', 'significance', 'loc_depth', 'sig_depth']

# 🌊 Load Flood CatBoost model
with open("catboost_model.pkl", "rb") as f:
    flood_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/earthquake')
def earthquake_form():
    return render_template('earthquake.html')

@app.route('/flood')
def flood_form():
    return render_template('flood.html')

@app.route('/predict_earthquake', methods=['POST'])
def predict_earthquake():
    try:
        tsunami = int(request.form['tsunami'])
        significance = float(request.form['significance'])
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        depth = float(request.form['depth'])
        year = int(request.form['year'])

        year_bin = pd.cut([year], bins=[1990, 2000, 2010, 2020, 2030], labels=False)[0]
        loc_depth = latitude * depth
        sig_depth = significance * depth

        input_dict = {
            'tsunami': tsunami,
            'significance': significance,
            'longitude': longitude,
            'latitude': latitude,
            'depth': depth,
            'year': year,
            'year_bin': year_bin,
            'loc_depth': loc_depth,
            'sig_depth': sig_depth
        }

        input_row = np.array([[input_dict[feat] for feat in eq_features]])
        scale_indices = [eq_features.index(col) for col in scale_cols]
        scaled_part = scaler.transform(input_row[:, scale_indices])
        input_row[0, scale_indices] = scaled_part

        prediction = earthquake_model.predict(input_row)[0]
        result = "⚠️ Earthquake Likely!" if prediction == 1 else "✅ No Earthquake Expected."

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error in Earthquake Prediction: {e}"

@app.route('/predict_flood', methods=['POST'])
def predict_flood():
    try:
        # 🌊 Collect all 20 input features
        features = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 'Urbanization',
            'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices', 'Encroachments',
            'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability', 'Landslides',
            'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors'
        ]

        input_data = {feat: float(request.form[feat]) for feat in features}
        input_df = pd.DataFrame([input_data])

        prediction = flood_model.predict(input_df)[0]
        result = f"🌊 Predicted Flood Severity: {round(prediction, 2)}"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error in Flood Prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
