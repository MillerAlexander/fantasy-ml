from flask import Flask, render_template_string
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create an instance of the Flask class to be our WSGI application
app = Flask(__name__)

# Basic HTML template to display our ML results on the web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fantasy AI - Web POC</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
        .container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .metric { font-size: 1.2em; color: #0066cc; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fantasy AI: Machine Learning Proof of Concept</h1>
        <p>This page dynamically fetched NFL data, trained a Random Forest Regressor on Wide Receiver stats, and evaluated the model.</p>
        <hr>
        <p><strong>Data Shape Processed:</strong> {{ data_shape }}</p>
        <p><strong>Model RMSE on Test Set:</strong> <span class="metric">{{ rmse }} PPR Points</span></p>
    </div>
</body>
</html>
"""

# Use the route() decorator to tell Flask what URL should trigger our function
@app.route('/')
def home():
    # 1. Fetch weekly data for the year specified
    df = nfl.import_weekly_data([2023])
    
    # 2. Process and filter
    wr_df = df[df['position'] == 'WR'].copy()
    features = ['targets', 'receptions', 'receiving_yards', 'receiving_tds']
    target = 'fantasy_points_ppr'
    
    model_df = wr_df.dropna(subset=features + [target])
    X = model_df[features]
    y = model_df[target]
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # 6. Serve to webpage
    return render_template_string(HTML_TEMPLATE, data_shape=model_df.shape, rmse=round(rmse, 2))

if __name__ == '__main__':
    # Run the web application on port 5000
    app.run(debug=True, port=5000)