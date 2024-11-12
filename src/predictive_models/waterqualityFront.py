import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model and scaler
model = load_model('C:/Users/KIIT/Desktop/AqualQ/src/predictive_models/water_quality_model.h5')
scaler = joblib.load('C:/Users/KIIT/Desktop/AqualQ/src/predictive_models/scaler1.pkl')

# Define AqualQ classification function
def classify_water_quality(pH):
    if 6.5 <= pH <= 8.5:
        return "Good"
    elif 5.5 <= pH < 6.5:
        return "Moderate"
    else:
        return "Poor"

# Streamlit layout
st.set_page_config(page_title="AqualQ Predictor", layout="wide")
st.title("🌊 AqualQ Predictor")
st.write("This app predicts pH levels and classifies AqualQ based on the predictions.")

# Sidebar with inputs
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.markdown("### Or enter feature values manually")
feature_values = {}

# Add sliders for each feature based on the column names except 'Date' and 'pH'
# Add sliders for each feature based on the column names except 'Date' and 'pH'
for feature in ["Chlorophyll", "Dissolved Oxygen", "Dissolved Oxygen Matter", "Suspended Matter", "Salinty", "Temperature", "Turbidity"]:
    feature_values[feature] = st.sidebar.slider(feature, min_value=-100.0, max_value=100.0, value=50.0)

# Load data and perform prediction if a file is uploaded or manual input is complete
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'])
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    
    # Preprocess data
    X = data.drop(columns=['Date', 'pH'])
    X_scaled = scaler.transform(X)
    
    # Predictions
    predictions = model.predict(X_scaled).flatten()
    
    # Add predictions and classification to the DataFrame
    data['Predicted pH'] = predictions
    data['AqualQ'] = data['Predicted pH'].apply(classify_water_quality)

    # Display results
    st.write("## Prediction Results")
    st.dataframe(data[['Date', 'Predicted pH', 'AqualQ']])

    # Download link for results
    csv = data.to_csv(index=False)
    st.download_button("Download Predictions", data=csv, file_name="predicted_water_quality.csv")

else:
    # Perform prediction on manually entered values
    X_manual = np.array([list(feature_values.values())]).reshape(1, -1)
    X_manual_scaled = scaler.transform(X_manual)
    prediction = model.predict(X_manual_scaled).flatten()[0]
    classification = classify_water_quality(prediction)

    st.write("## Prediction Result")
    st.metric(label="Predicted pH", value=f"{prediction:.2f}")
    st.metric(label="AqualQ Classification", value=classification)

# Colorful design with water-themed tones
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
