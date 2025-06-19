def test_predict_with_correct_features():
    from ml_app.api import predict
    import pandas as pd

    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    response = predict(df)
    assert response is not None

def test_predict_with_unseen_features():
    from ml_app.api import predict
    import pandas as pd

    df = pd.DataFrame({'incident': [1, 2]})
    try:
        predict(df)
    except ValueError as e:
        assert str(e) == "The feature names should match those that were passed during fit."