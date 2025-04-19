# ----------------------------------------
# Imports
# ----------------------------------------

import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------------
# Configuration
# ----------------------------------------

DATA_PATH_X = "/Users/dhanu/Downloads/X_lstm_advanced.npy"
DATA_PATH_Y = "/Users/dhanu/Downloads/y_lstm_advanced.npy"

TRAIN_RATIO = 0.8
BATCH_SIZE = 16
EPOCHS = 60
PATIENCE = 7


# ----------------------------------------
# Environment Setup
# ----------------------------------------

def log_environment():
    print("=" * 60)
    print(" [INFO] Starting LSTM Training Script")
    print(" [INFO] NumPy version:", np.__version__)
    print(" [INFO] TensorFlow version:", tf.__version__)
    print(" [INFO] Timestamp:", time.ctime())
    print(" [INFO] Working Directory:", os.getcwd())
    print("=" * 60)


# ----------------------------------------
# Data Loading
# ----------------------------------------

def load_data(path_x, path_y):
    print("[INFO] Loading data...")
    X = np.load(path_x)
    y = np.load(path_y)
    print("[INFO] Data loaded successfully!")
    return X, y


# ----------------------------------------
# Data Inspection
# ----------------------------------------

def describe_data(X, y):
    print("\n[INFO] Dataset Overview:")
    print("-" * 50)
    print(" Shape of X:", X.shape)
    print(" Shape of y:", y.shape)
    print(" Number of samples:", len(X))
    print(" Sequence length:", X.shape[1])
    print(" Feature dimension:", X.shape[2])
    print("-" * 50)


# ----------------------------------------
# Data Splitting
# ----------------------------------------

def split_data(X, y, ratio):
    print("[INFO] Splitting data into train and test sets...")
    split_index = int(len(X) * ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    print("[INFO] Data split complete!")
    print(f" - Training samples: {len(X_train)}")
    print(f" - Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ----------------------------------------
# Model Architecture
# ----------------------------------------

def build_lstm_model(input_shape):
    print("[INFO] Building the LSTM model...")

    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("[INFO] Model built successfully!\n")
    model.summary()

    return model


# ----------------------------------------
# Callbacks
# ----------------------------------------

def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )
    return [early_stop]


# ----------------------------------------
# Training Function
# ----------------------------------------

def train_model(model, X_train, y_train):
    print("\n[INFO] Starting model training...\n")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=1
    )
    print("\n[INFO] Training complete!")
    return history


# ----------------------------------------
# Evaluation Function
# ----------------------------------------

def evaluate_model(model, X_test, y_test):
    print("\n[INFO] Evaluating model on test data...\n")
    y_probs = model.predict(X_test)
    y_pred = (y_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(" Accuracy:", acc)
    print(" Confusion Matrix:\n", cm)
    print(" Classification Report:\n", cr)


# ----------------------------------------
# Helper Utility (for future extension)
# ----------------------------------------

def save_model(model, filename="lstm_model.h5"):
    print(f"[INFO] Saving model to '{filename}'...")
    model.save(filename)
    print("[INFO] Model saved successfully!")


# ----------------------------------------
# Main
# ----------------------------------------

def main():
    log_environment()

    X, y = load_data(DATA_PATH_X, DATA_PATH_Y)
    describe_data(X, y)

    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_RATIO)

    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_model(input_shape)

    history = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # Optional: Uncomment to save model
    # save_model(model)


# ----------------------------------------
# Entry Point
# ----------------------------------------

if __name__ == "__main__":
    main()
