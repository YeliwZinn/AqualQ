import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load Data Function
def load_data(file_path):
    """Loads the dataset and performs initial preprocessing."""
    data = pd.read_csv("C:/Users/KIIT/Desktop/AqualQ/src/deployment/merged_data.csv")
    
    # Convert Date column to datetime, if it exists
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    
    # Drop any rows with missing values
    data = data.dropna()
    
    return data

# Prepare Data for Modeling
def prepare_data(data, target_column, feature_columns):
    """Splits data into training and testing sets and scales features."""
    X = data[feature_columns]
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Train Model
def train_model(X_train, y_train):
    """Trains a Random Forest Regressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and returns the error metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return mae, rmse

# Streamlit App Interface
st.title("Environmental Metric Prediction")
st.write("This app allows you to train a predictive model and make predictions for environmental metrics.")

# Load Data
data_path = os.path.join("..", "data", "your_dataset.csv")  # Adjust path if necessary
data = load_data(data_path)

# Define columns
target_column = "Dissolved Oxygen"  # Replace with the metric you want to predict
feature_columns = ["Chlorophyll", "Temperature", "Salinty", "Turbidity", "pH"]  # Adjust as needed

# Train Model Button
if st.button("Train Predictive Model"):
    X_train, X_test, y_train, y_test, scaler = prepare_data(data, target_column, feature_columns)
    model = train_model(X_train, y_train)
    
    # Save the model and scaler
    model_save_path = os.path.join("..", "predictive_model.joblib")
    scaler_save_path = os.path.join("..", "scaler.joblib")
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    st.success("Model trained and saved successfully.")
    
    # Evaluate model
    mae, rmse = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Prediction Section
st.header("Make a Prediction")
st.write("Use the sliders to input values for each feature, then click 'Predict'.")

# Input sliders for each feature
input_data = []
for feature in feature_columns:
    value = st.slider(f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
    input_data.append(value)

if st.button("Predict"):
    # Load model and scaler
    model = joblib.load(os.path.join("..",  "predictive_model.joblib"))
    scaler = joblib.load(os.path.join("..", "scaler.joblib"))
    
    # Prepare input data for prediction
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted {target_column}: {prediction[0]}")
