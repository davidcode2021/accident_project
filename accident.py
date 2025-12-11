from flask import Flask, request, render_template
from feature_engineering import FeatureEngineering, TargetEncoder
import numpy as np
import pandas as pd 
import scipy
import joblib
import sys
import os
import requests


# print("=== DIAGNOSTIC: INSTALLED VERSIONS ===", file=sys.stderr)
# print(f"Python: {sys.version}", file=sys.stderr)
# print(f"NumPy: {numpy.__version__}", file=sys.stderr)
# print(f"pandas: {pandas.__version__}", file=sys.stderr)
# print(f"xgboost: {xgboost.__version__}", file=sys.stderr)
# print(f"scikit-learn: {sklearn.__version__}", file=sys.stderr)
# print(f"joblib: {joblib.__version__}", file=sys.stderr)
# print(f"scipy: {scipy.__version__}", file=sys.stderr)
# print("======================================", file=sys.stderr)


MODEL_URL = "https://drive.google.com/uc?id=1HUpw8rhbi4BAiCWTzsVHq-oyBcOy6hDh&export=download"  # replace with your file ID

if not os.path.exists("model_pipeline.pkl"):
    print("Downloading model from Google Drive...")
    r = requests.get(MODEL_URL)
    with open("model_pipeline.pkl", "wb") as f:
        f.write(r.content)
    

sys.modules['__main__'].FeatureEngineer = FeatureEngineering
sys.modules['__main__'].TargetEncoder = TargetEncoder

model=joblib.load("model_pipeline.pkl")


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
