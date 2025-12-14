# accident_project
model to predict the rate of accident 
# Accident Risk Prediction Pipeline

## Overview
This project predicts road accident risk using structured and engineered features. It combines a **weighted prior risk score**, **feature engineering**, **smoothed target encoding**, and **XGBoost regression** to estimate risk probabilities and residual corrections.

**Data Source:**  
The dataset used in this project is obtained from **Kaggle**.


The pipeline is modular, allowing:

- Structured risk modeling (`y_structured`)
- Feature engineering (categorical and numeric features)
- Target encoding for categorical and combined features
- Residual prediction using XGBoost
- 
## Model Storage (Google Drive)

### Why the model is not in the repository
The trained model file (`model_pipeline.pkl`) is relatively large and exceeds GitHub’s recommended file size limits.  
To keep the repository lightweight and avoid Git LFS complexity, the model is stored externally on **Google Drive**.

---

### Model Location
The model is hosted on Google Drive and made publicly accessible (read-only).

**Google Drive file link:**  
https://drive.google.com/file/d/1HUpw8rhbi4BAiCWTzsVHq-oyBcOy6hDh/view

**Google Drive File ID:**  
1HUpw8rhbi4BAiCWTzsVHq-oyBcOy6hDh


---

## Features

1. **Feature Engineering**
   - Boolean features: `is_low_visibility`, `bad_weather`, `rush_hour`
   - Numeric features: `sharp_curve`, `high_speed`, `speed_curvature`, `curve_dark`, `curvature_group`
   - Factorization of categorical variables for numeric model input

2. **Target Encoding**
   - Smoothed mean encoding of categorical columns and combinations
   - Prevents overfitting small categories with configurable smoothing parameter

3. **Residual Modeling**
   - Predicts corrections on top of a structured prior (`y_structured`)
   - Final predictions: `predicted_risk = y_structured + residual_prediction`

4. **Model**
   - XGBoost regressor
   - Configured with tuned hyperparameters and trained on engineered features + TE
---

### Clone the repository

```bash
git clone gh repo clone davidcode2021/accident_project

```
