"""Modular ML pipeline for the Churn Modelling dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Resolve paths relative to this file, so the code works from any directory in WSL
BASE_DIR = Path(__file__).resolve().parent
# Linux (WSL) is case-sensitive: this must match the real file name exactly
DEFAULT_DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "model.joblib"
DEFAULT_SCALER_PATH = BASE_DIR / "scaler.joblib"


def load_data(data_path=DEFAULT_DATA_PATH):
    """Read the CSV file and return a DataFrame."""
    return pd.read_csv(data_path)


def drop_useless_columns(data, columns=("Surname", "Geography")):
    """Drop columns that are not useful for prediction."""
    return data.drop(list(columns), axis=1)


def encode_gender(data):
    """Encode Gender as numbers (Female -> 0, Male -> 1). Returns a new DataFrame."""
    data = data.copy()
    data["Gender"] = LabelEncoder().fit_transform(data["Gender"])
    return data


def remove_missing_values(data):
    """Remove rows that contain missing values."""
    return data.dropna()


def split_features_target(data):
    """Separate the features (X) from the target (y = Exited)."""
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=1):
    """Split into train and test sets (80% / 20% by default)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(x_train, x_test, scaler_path=DEFAULT_SCALER_PATH):
    """Fit a StandardScaler on the train set only, apply it to both sets,
    and save the scaler for later use (inference)."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, scaler_path)
    return x_train_scaled, x_test_scaled


def prepare_data(data_path=DEFAULT_DATA_PATH, scaler_path=DEFAULT_SCALER_PATH):
    """Run the whole preparation pipeline.

    Returns:
        x_train_scaled, x_test_scaled, y_train, y_test
    """
    data = load_data(data_path)
    data = drop_useless_columns(data)
    data = encode_gender(data)
    data = remove_missing_values(data)
    X, y = split_features_target(data)
    x_train, x_test, y_train, y_test = split_train_test(X, y)
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test, scaler_path)
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train):
    """Train a RandomForest classifier and return it."""
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Compute and print the accuracy of the model on the test set."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, model_path=DEFAULT_MODEL_PATH):
    """Save the trained model to disk with joblib."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load a saved model from disk."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict(features, model_path=DEFAULT_MODEL_PATH):
    """Predict with the saved model. `features` must already be scaled."""
    model = joblib.load(model_path)
    return model.predict(features)
