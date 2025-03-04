import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Dummy data setup for testing
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.rand(1000, 1)   # Regression target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the Transformer-based model
def build_transformer_model(input_shape, embed_dim=32, num_heads=2, ff_dim=32, rate=0.1):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(embed_dim, activation="relu")(inputs)  # Input embedding

    # Use a Keras layer for reshaping instead of tf.expand_dims
    x = layers.Reshape((1, embed_dim))(x)  # Convert to 3D for MultiHeadAttention

    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(x)  # Transformer block

    # Flatten before passing to output
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)  # Output layer (for regression)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
input_shape = (X_train.shape[1],)
model = build_transformer_model(input_shape)
model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

print("X_train shape:", X_train.shape)
# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)  # Adjusted epochs for testing