import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.losses import mean_squared_error as mse

# Load the model... (rest of your code)
# Load the saved model and scalers
model = load_model('C:/Users/KIIT/Desktop/AqualQ/src/predictive_models/lstm_model.h5')
scaler_features = joblib.load('C:/Users/KIIT/Desktop/AqualQ/src/predictive_models/scaler_features.pkl')
scaler_target = joblib.load('C:/Users/KIIT/Desktop/AqualQ/src/predictive_models/scaler_target.pkl')

# Define sequence creation function
def create_sequences(features, time_steps=30):
    X = []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
    return np.array(X)

# Load and preprocess uploaded dataset
def preprocess_data(data, features, scaler_features, scaler_target, time_steps=30):
    # Sort data and fill missing values
    data = data.sort_values('Date').fillna(method='ffill').dropna()
    scaled_features = scaler_features.transform(data[features])
    
    # Create sequences
    X = create_sequences(scaled_features, time_steps)
    return X, data['Date'][time_steps:], data

# App layout and design
st.set_page_config(page_title="AqualQ Predictor - LSTM Model", layout="wide")
st.title("🌊 LSTM AqualQ Predictor")
st.write("This app uses an LSTM model to predict pH levels in water and classifies the AqualQ based on predictions.")

# Sidebar for user input
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Define model parameters for sequence prediction
time_steps = 30
features = ['Chlorophyll', 'Dissolved Oxygen', 'Dissolved Oxygen Matter', 'Suspended Matter', 'Salinty', 'Temperature', 'Turbidity']
target = 'pH'

# Main application logic
if uploaded_file:
    # Load and preprocess data
    data = pd.read_csv(uploaded_file, parse_dates=['Date'])
    X, dates, full_data = preprocess_data(data, features, scaler_features, scaler_target, time_steps)
    
    # Predict and scale back to original values
    predictions = model.predict(X)
    predicted_pH = scaler_target.inverse_transform(predictions).flatten()

    # Select actual pH values for the same period
    actual_pH = full_data[target][time_steps:].values

    # Prepare results dataframe
    results = pd.DataFrame({
        'Date': dates.reset_index(drop=True),
        'Actual pH': actual_pH,
        'Predicted pH': predicted_pH
    })

    # Display results and allow download
    st.write("## Prediction Results")
    st.dataframe(results)
    
    csv = results.to_csv(index=False)
    st.download_button("Download Predictions", data=csv, file_name="lstm_water_quality_predictions.csv")

    # Plot predictions vs. actual values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['Date'], y=results['Actual pH'], mode='lines+markers', name='Actual pH', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=results['Date'], y=results['Predicted pH'], mode='lines+markers', name='Predicted pH', line=dict(color='cyan')))
    fig.update_layout(title="Actual vs. Predicted pH", xaxis_title="Date", yaxis_title="pH Level", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Upload a dataset in CSV format to view predictions.")

# Colorful water-themed design
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
        }
        .stApp {
            color: #007acc;
        }
        .stButton>button {
            color: #007acc;
            background-color: #b3e5fc;
            border: 2px solid #007acc;
        }
        .stDownloadButton>button {
            color: #004d73;
            background-color: #e1f5fe;
            border: 2px solid #004d73;
        }
        .stMetricValue {
            font-size: 25px;
            font-weight: bold;
            color: #004d73;
        }
    </style>
""", unsafe_allow_html=True)
