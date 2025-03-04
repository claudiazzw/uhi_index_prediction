######################################
# 1. IMPORTS
######################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

######################################
# 2. DATA PREPARATION
######################################
# Suppose uhi_data_modified is already in your workspace
# We'll print the head to confirm structure
print("Head of uhi_data_modified:")
display(uhi_data_modified.head())

# Separate features (X) and target (y)
X = uhi_data_modified.drop(columns=["UHI Index"]).values  # shape: (n_samples, 6)
y = uhi_data_modified["UHI Index"].values                # shape: (n_samples,)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for 1D CNN:
# We add a "channels" dimension so that our input becomes (batch_size, num_features, 1)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

print("\nTraining features shape:", X_train_cnn.shape)
print("Testing features shape:", X_test_cnn.shape)

######################################
# 3. MODEL ARCHITECTURE (1D CNN)
######################################
def build_cnn_model(input_shape):
    """
    input_shape: (num_features, 1)
    """
    inputs = keras.Input(shape=input_shape)
    
    # First Conv block
    x = layers.Conv1D(filters=32, kernel_size=2, activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Second Conv block
    x = layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Dense block
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer for regression
    outputs = layers.Dense(1, activation="linear")(x)
    
    model = keras.Model(inputs, outputs)
    return model

######################################
# 4. COMPILE & TRAIN THE MODEL
######################################
# Build model
cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
cnn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Show summary
cnn_model.summary()

# Train
history = cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.2,
    epochs=50,        # You can increase to 100+ for better performance
    batch_size=32,
    verbose=1
)

######################################
# 5. EVALUATE THE MODEL
######################################
test_loss, test_mae = cnn_model.evaluate(X_test_cnn, y_test, verbose=1)
print("\nTest MSE (loss):", test_loss)
print("Test MAE:", test_mae)

print("X_train shape:", X_train.shape)

# Example predictions
preds = cnn_model.predict(X_test_cnn[:5])
print("\nSample predictions:", preds.flatten())
print("Actual values:", y_test[:5])