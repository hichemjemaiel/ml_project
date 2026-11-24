import argparse
import subprocess  # nosec B404
import sys

import pandas as pd
from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

# Only scan our own files (not venv/, which would take forever)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
MODEL_PATH = "model.joblib"


def run(args):
    """Run a command with the current Python interpreter (the venv one)."""
    subprocess.run([sys.executable, "-m", *args], check=True)  # nosec B603


# ---------------------------------------------------------------- TASKS (env)
@task(name="install")
def install_task():
    run(["pip", "install", "-r", "requirements.txt"])


# ----------------------------------------------------------- TASKS (code)
@task(name="format")
def format_task():
    run(["black", *FILES])


@task(name="quality")
def quality_task():
    run(["flake8", *FILES, "--max-line-length=100"])


@task(name="security")
def security_task():
    # -ll : only report medium and high severity issues
    run(["bandit", "-ll", *FILES])


@task(name="unit-tests")
def test_task():
    run(["pytest", "tests/", "-v"])


# ------------------------------------------------------- TASKS (data/model)
@task(name="prepare")
def prepare_task():
    return prepare_data()  # X_train, X_test, y_train, y_test


@task(name="train")
def train_task(data):
    X_train, X_test, y_train, y_test = data
    return train_model(X_train, y_train)


@task(name="save")
def save_task(model):
    save_model(model, MODEL_PATH)


@task(name="load")
def load_task():
    return load_model(MODEL_PATH)


@task(name="evaluate")
def evaluate_task(model, data):
    _, X_test, _, y_test = data
    return evaluate_model(model, X_test, y_test)


@task(name="predict")
def predict_task(model, data):
    _, X_test, _, _ = data
    predictions = model.predict(X_test)
    pd.DataFrame({"prediction": predictions}).to_csv("predictions.csv", index=False)
    return predictions


# ---------------------------------------------------------------------- FLOWS
@flow(name="install")
def install_flow():
    install_task()


@flow(name="code")
def code_flow():
    install_task()
    format_task()
    quality_task()
    security_task()
    test_task()


@flow(name="train")
def train_flow():
    data = prepare_task()
    model = train_task(data)
    save_task(model)  # needed so the weekly retraining is used by "predict"
    return model


@flow(name="evaluate")
def evaluate_flow():
    data = prepare_task()
    model = load_task()
    return evaluate_task(model, data)


@flow(name="predict")
def predict_flow():
    data = prepare_task()
    model = load_task()
    return predict_task(model, data)


@flow(name="all")
def all_flow():
    install_task()
    format_task()
    quality_task()
    security_task()
    test_task()
    data = prepare_task()
    model = train_task(data)
    save_task(model)
    evaluate_task(model, data)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
    "predict": predict_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
