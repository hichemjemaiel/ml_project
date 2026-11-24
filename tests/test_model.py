from model_pipeline import prepare_data


def test_prepare_data_returns_four_non_empty_parts():
    data = prepare_data()
    assert len(data) == 4  # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = data
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
