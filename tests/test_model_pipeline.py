import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


# ------------------------------------------------------------------ FIXTURES
# scope="module": data and model are built once and reused by all tests (faster)
@pytest.fixture(scope="module")
def data():
    return prepare_data()  # X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def model(data):
    X_train, _, y_train, _ = data
    return train_model(X_train, y_train)


# -------------------------------------------------------------- prepare_data
def test_prepare_data_returns_four_parts(data):
    assert len(data) == 4


def test_prepare_data_shapes_are_consistent(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(X_test)  # train set is the bigger split


def test_prepare_data_same_number_of_features(data):
    X_train, X_test, _, _ = data
    assert np.asarray(X_train).shape[1] == np.asarray(X_test).shape[1]


def test_prepare_data_has_no_missing_values(data):
    X_train, X_test, y_train, y_test = data
    for part in (X_train, X_test, y_train, y_test):
        assert pd.DataFrame(part).isnull().sum().sum() == 0


def test_prepare_data_target_is_binary(data):
    _, _, y_train, y_test = data
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}


# --------------------------------------------------------------- train_model
def test_train_model_returns_model_with_predict(model):
    assert hasattr(model, "predict")


def test_train_model_predictions_have_right_length(model, data):
    _, X_test, _, y_test = data
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_model_predictions_are_binary(model, data):
    _, X_test, _, _ = data
    assert set(np.unique(model.predict(X_test))) <= {0, 1}


# ------------------------------------------------------- save_model/load_model
def test_save_model_creates_file(model, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_then_load_gives_same_predictions(model, data, tmp_path):
    _, X_test, _, _ = data
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    loaded = load_model(str(path))
    np.testing.assert_array_equal(model.predict(X_test), loaded.predict(X_test))


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        load_model(str(tmp_path / "does_not_exist.joblib"))


# ------------------------------------------------------------- evaluate_model
def test_evaluate_model_reports_accuracy(model, data, capsys):
    _, X_test, _, y_test = data
    result = evaluate_model(model, X_test, y_test)
    printed = capsys.readouterr().out
    returned_ok = isinstance(result, (int, float)) and 0 <= result <= 1
    assert "ccuracy" in printed or returned_ok


def test_model_accuracy_is_reasonable(model, data):
    _, X_test, _, y_test = data
    accuracy = (model.predict(X_test) == np.asarray(y_test)).mean()
    assert accuracy >= 0.70  # your pipeline reached ~0.85
