import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the data
data = pd.read_csv('src/deployment/merged_data.csv', parse_dates=['Date'])

# Handle missing values and drop remaining NA values
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)

# Sort by date to ensure chronological order
data.sort_values('Date', inplace=True)

# Select features and target
X = data.drop(columns=['Date', 'pH'])
y = data['pH']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output for pH
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, 
                    validation_split=0.2, epochs=100, 
                    batch_size=32, callbacks=[early_stopping])

# Save the trained model and scaler
model.save('src/predictive_models/water_quality_model.h5')
scaler_filename = 'src/predictive_models/scaler1.pkl'
joblib.dump(scaler, scaler_filename)

# Predict on the test data
predictions = model.predict(X_test_scaled)

# Align dates with test set and save results
results = pd.DataFrame({
    'Date': data['Date'].iloc[y_test.index].values,
    'Actual pH': y_test.values,
    'Predicted pH': predictions.flatten()
})
results.to_csv('src/predictive_models/waaterqpred.csv', index=False)

# Function to classify AqualQ based on pH value
def classify_water_quality(pH):
    if 6.5 <= pH <= 8.5:
        return "Good"
    elif 5.5 <= pH < 6.5:
        return "Moderate"
    else:
        return "Poor"

# Classify predictions
predicted_classes = [classify_water_quality(p) for p in predictions.flatten()]

print("Model training complete, predictions saved, and classification performed.")
