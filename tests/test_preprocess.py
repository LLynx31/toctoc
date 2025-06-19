def test_preprocess_uploaded_df_correct_columns():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from ml_app.data.preprocess import preprocess_uploaded_df

    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    columns = ['feature1', 'feature2']
    scaler = StandardScaler()
    scaler.fit(df[columns])

    X_pred, to_predict = preprocess_uploaded_df(df.copy(), columns, scaler)
    assert X_pred.shape == df.shape
    assert to_predict.equals(df[columns])

def test_preprocess_uploaded_df_unseen_columns():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from ml_app.data.preprocess import preprocess_uploaded_df

    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'incident': [4, 5, 6]
    })
    columns = ['feature1']
    scaler = StandardScaler()
    scaler.fit(df[columns])

    try:
        preprocess_uploaded_df(df.copy(), columns, scaler)
    except ValueError as e:
        assert str(e) == "The feature names should match those that were passed during fit."