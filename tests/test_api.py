def test_predict_with_correct_features(client):
    response = client.post("/predict", json={"data": {"feature1": 1, "feature2": 2}})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_with_unseen_features(client):
    response = client.post("/predict", json={"data": {"feature1": 1, "feature2": 2, "incident": 3}})
    assert response.status_code == 400
    assert response.json()["detail"] == "Feature names unseen at fit time: ['incident']"