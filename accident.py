from flask import Flask, request, render_template
from feature_engineering import FeatureEngineering, TargetEncoder
import numpy as np
import pandas as pd 
import scipy
import joblib
import sys
import os
import requests

MODEL_PATH = "model_pipeline.pkl"
GDRIVE_ID = "1HUpw8rhbi4BAiCWTzsVHq-oyBcOy6hDh"

def download_from_gdrive(file_id, destination):
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    download_from_gdrive(GDRIVE_ID, MODEL_PATH)

print("Model size:", os.path.getsize(MODEL_PATH))

  
sys.modules['__main__'].FeatureEngineering = FeatureEngineering
sys.modules['__main__'].TargetEncoder = TargetEncoder

model=joblib.load(MODEL_PATH)
app = Flask(__name__)

# home page
@app.route("/")
def home():
    return render_template("accident.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "road_type": request.form.get("road_type"),
        "num_lanes": int(request.form.get("num_lanes")),
        "curvature": float(request.form.get("curvature")),
        "speed_limit": int(request.form.get("speed_limit")),
        "lighting": request.form.get("lighting"),
        "weather": request.form.get("weather"),
        "road_signs_present": "road_signs_present" in request.form,
        "public_road": "public_road" in request.form,
        "time_of_day": request.form.get("time_of_day"),
        "holiday": "holiday" in request.form,
        "school_season": "school_season" in request.form,
        "num_reported_accidents": int(request.form.get("num_reported_accidents")),
    }
    X = pd.DataFrame([data])
    
    def f(X):
        return \
        0.3 * X["curvature"] + \
        0.2 * (X["lighting"] == "night").astype(int) + \
        0.1 * (X["weather"] != "clear").astype(int) + \
        0.2 * (X["speed_limit"] >= 60).astype(int) + \
        0.1 * (X["num_reported_accidents"] > 2).astype(int)

    def clip(f):
        def clip_f(X):
            sigma = 0.05
            mu = f(X)
            a, b = -mu/sigma, (1-mu)/sigma
            Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
            phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
            return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
        return clip_f

    z = clip(f)(X)[0]
    X["y_structured"] = z
    # reconstructing our true predicted value 
    prediction = model.predict(X)[0] + z
    risk = float(prediction)

    return render_template("accident.html", risk=risk)
        

if __name__ == "__main__":
    app.run(debug=True)
