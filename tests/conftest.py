import pytest

@pytest.fixture
def sample_dataframe():
    import pandas as pd
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'incident': [7, 8, 9]
    })

@pytest.fixture
def scaler():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit([[1], [2], [3]])
    return scaler

@pytest.fixture
def valid_columns():
    return ['feature1', 'feature2']

@pytest.fixture
def invalid_columns():
    return ['feature1', 'feature2', 'unseen_feature']

@pytest.mark.parametrize("columns, expected", [
    (['feature1', 'feature2'], True),
    (['feature1', 'feature2', 'unseen_feature'], False)
])
def test_predict_function(sample_dataframe, scaler, columns, expected):
    from ml_app.api import predict  # Adjust the import based on your actual structure
    if expected:
        result = predict(sample_dataframe, columns, scaler)
        assert result is not None  # Replace with actual expected result check
    else:
        with pytest.raises(ValueError):
            predict(sample_dataframe, columns, scaler)