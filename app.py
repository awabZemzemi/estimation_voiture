"""
Simple Flask Web App for Car Price Estimation
"""
import flask
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os

app = Flask(__name__)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

# Load the cleaned data to get column names and categorical info
df_clean = pd.read_csv('cars_data_2.csv')

# Drop unnecessary columns (same as in notebook)
columns_to_drop = [
    'listing_id', 'url', 'listing_date',  'price_rating','version','engine',
    'interior_color', 'exterior_color', 'upholstery', 'power_ch', 'cylinder_capacity', 'location'
]
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

# Data preprocessing (same as notebook)
df_clean['fuel_type'].fillna(df_clean['fuel_type'].mode()[0], inplace=True)
df_clean['car_condition'].fillna(df_clean['car_condition'].mode()[0], inplace=True)
df_clean['previous_owners'].fillna(df_clean['previous_owners'].mode()[0], inplace=True)
df_clean['body_type'].fillna(df_clean['body_type'].mode()[0], inplace=True)
df_clean['seats'].fillna(df_clean['seats'].median(), inplace=True)
df_clean['doors'].fillna(df_clean['doors'].median(), inplace=True)

# Feature engineering
df_clean['is_showroom'] = (df_clean['seller'] != 'Individual').astype(int)
df_clean['age'] = 2025 - df_clean['year']
df_clean = df_clean.drop(columns=['year'])

condition_map = {
    'Très bon': 3,
    'Normal': 2,
    'Moyen': 1,
    'En panne / Accidenté': 0
}
df_clean['car_condition'] = df_clean['car_condition'].map(condition_map).astype('Int64')

age_safe = df_clean['age']
miles = df_clean['mileage']
df_clean['miles_per_year'] = miles.where(age_safe <= 1, miles / age_safe)
df_clean = df_clean.drop(columns=['mileage', 'seller'])

# Additional features (same as notebook)
df_clean["age_squared"] = df_clean["age"] ** 2
df_clean["miles_per_year_squared"] = df_clean["miles_per_year"] ** 2
df_clean["condition_age_ratio"] = df_clean["car_condition"] / (df_clean["age"].replace(0, 1))

# Extract first digit from previous_owners
df_clean['previous_owners'] = df_clean['previous_owners'].astype(str).str.strip().str[0]

# Store categorical columns
CAT_COLS = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
FEATURE_COLUMNS = df_clean.drop('price', axis=1).columns.tolist()

# Train a model or load existing one
print("Training model...")
X = df_clean.drop('price', axis=1)
y = df_clean['price']

MODEL = CatBoostRegressor(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    subsample=0.85,
    rsm=0.85,
    min_data_in_leaf=10,
    grow_policy="SymmetricTree",
    loss_function="MAE",
    cat_features=CAT_COLS,
    verbose=0,
    random_state=42
)
MODEL.fit(X, y)
print("✓ Model trained successfully!")

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/get-options', methods=['GET'])
def get_options():
    """Get categorical options for dropdowns"""
    options = {}
    for col in CAT_COLS:
        options[col] = sorted(df_clean[col].unique().tolist())
    return jsonify(options)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.json
        
        # Create input dataframe
        input_data = {}
        for col in FEATURE_COLUMNS:
            if col in data:
                # Keep categorical columns as strings, convert numeric to float
                if col in CAT_COLS:
                    input_data[col] = str(data[col])
                else:
                    try:
                        input_data[col] = float(data[col])
                    except:
                        input_data[col] = data[col]
            else:
                # Use default value from training data
                if col in CAT_COLS:
                    input_data[col] = str(df_clean[col].mode()[0])
                else:
                    input_data[col] = df_clean[col].median()
        
        # Convert to dataframe
        X_input = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = MODEL.predict(X_input)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'currency': 'TND'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'CatBoost Regressor',
        'total_features': len(FEATURE_COLUMNS),
        'training_samples': len(df_clean),
        'categorical_features': len(CAT_COLS),
        'numeric_features': len(FEATURE_COLUMNS) - len(CAT_COLS)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 CAR PRICE ESTIMATION - WEB INTERFACE")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
