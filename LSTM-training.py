# ---------------------------------------------
# Import Required Libraries
# ---------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
import time


# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

def log_data_info(X, y):
    print("[INFO] Shape of X:", X.shape)
    print("[INFO] Shape of y:", y.shape)
    print("[INFO] Data type of X:", type(X))
    print("[INFO] Data type of y:", type(y))
    print("[INFO] Number of samples:", len(X))
    print("-" * 50)


def split_data(X, y, train_ratio=0.8):
    print("[INFO] Splitting data into train and test sets...")
    split_index = int(len(X) * train_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    print("[INFO] Split completed. Train size:", len(X_train), "Test size:", len(X_test))
    return X_train, X_test, y_train, y_test


def build_model(input_shape):
    print("[INFO] Building LSTM model...")
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("[INFO] Model built successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    print("[INFO] Predicting on test data...")
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("\n[RESULT] Accuracy:", acc)
    print("\n[RESULT] Confusion Matrix:\n", cm)
    print("\n[RESULT] Classification Report:\n", cr)


def train_model(model, X_train, y_train):
    print("[INFO] Starting model training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    print("[INFO] Training completed.")
    return history


def setup_environment():
    print("[INFO] TensorFlow version:", tf.__version__)
    print("[INFO] NumPy version:", np.__version__)
    print("[INFO] Current Working Directory:", os.getcwd())
    print("[INFO] Timestamp:", time.ctime())
    print("-" * 50)


# ---------------------------------------------
# Main Execution
# ---------------------------------------------

def main():
    setup_environment()

    print("[INFO] Loading preprocessed data...")
    X = np.load("/Users/dhanu/Downloads/X_lstm_balanced.npy")
    y = np.load("/Users/dhanu/Downloads/y_lstm_balanced.npy")

    log_data_info(X, y)

    X_train, X_test, y_train, y_test = split_data(X, y)

    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape)

    history = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)


# ---------------------------------------------
# Script Entry Point
# ---------------------------------------------

if __name__ == "__main__":
    main()
