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

# 🔁 Load model, scaler, and feature names
with open("earthquake_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Features that need to be scaled
scale_cols = ['longitude', 'latitude', 'depth', 'significance', 'loc_depth', 'sig_depth']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🌐 Collect data from form
        tsunami = int(request.form['tsunami'])
        significance = float(request.form['significance'])
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        depth = float(request.form['depth'])
        year = int(request.form['year'])

        # 🎯 Feature engineering
        year_bin = pd.cut([year], bins=[1990, 2000, 2010, 2020, 2030], labels=False)[0]
        loc_depth = latitude * depth
        sig_depth = significance * depth

        # 🎁 Build input
        user_input_dict = {
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

        input_row = np.array([[user_input_dict[feat] for feat in feature_names]])

        # 🧼 Scale specific features
        scale_indices = [feature_names.index(col) for col in scale_cols]
        scaled_part = scaler.transform(input_row[:, scale_indices])
        input_row[0, scale_indices] = scaled_part

        # ✅ Predict
        prediction = model.predict(input_row)[0]
        result = "⚠️ Earthquake Likely!" if prediction == 1 else "✅ No Earthquake Expected."

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
