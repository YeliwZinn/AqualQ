import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
data = pd.read_csv('src/deployment/merged_data.csv', parse_dates=['Date'])

# Convert 'Date' column to datetime format (already handled by parse_dates)
data['Date'] = pd.to_datetime(data['Date'])

# Fill missing values using forward fill
data.fillna(method='ffill', inplace=True)

# Drop any remaining rows with missing values (in case of gaps at the beginning)
data.dropna(inplace=True)

# Sort data by date to ensure sequential order
data.sort_values('Date', inplace=True)

# Select features and target column
features = ['Chlorophyll', 'Dissolved Oxygen', 'Dissolved Oxygen Matter', 'Suspended Matter', 'Salinty', 'Temperature', 'Turbidity']
target = 'pH'

# Scale features and target
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit scalers on the data and transform
scaled_features = scaler_features.fit_transform(data[features])
scaled_target = scaler_target.fit_transform(data[[target]])

# Function to create sequences
def create_sequences(features, target, time_steps=30):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# Set the sequence length (time_steps) and create sequences
time_steps = 30
X, y = create_sequences(scaled_features, scaled_target, time_steps)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32, return_sequences=False),
    Dense(1)  # Predicting a single value (next day's pH)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=50, 
                    batch_size=32, 
                    callbacks=[early_stopping])

# Save the trained model and scalers
model.save('src/predictive_models/lstm_model.h5')
joblib.dump(scaler_features, 'src/predictive_models/scaler_features.pkl')
joblib.dump(scaler_target, 'src/predictive_models/scaler_target.pkl')

# Test the model
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predicted_pH = scaler_target.inverse_transform(predictions.reshape(-1, 1))
actual_pH = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Save predictions for visualization
results = pd.DataFrame({
    'Date': data['Date'].iloc[train_size + time_steps:].reset_index(drop=True),
    'Actual pH': actual_pH.flatten(),
    'Predicted pH': predicted_pH.flatten()
})
results.to_csv('src/predictive_models/lstm_predictions.csv', index=False)

print("LSTM Model Training and Testing Complete.")
