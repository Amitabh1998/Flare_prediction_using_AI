import numpy as np
from app.model import train_all_models, predict_flare
from app.data import load_data

def test_model_inference():
    df = load_data()
    features = ['total_sleep', 'deep_sleep', 'rem_sleep', 'wake', 'pain', 'fatigue', 'mood', 'sleep_hours', 'treatment']
    model, model_type, _, _, mean, std = train_all_models(df, features, 'models/test_model.pt')
    input_data = np.random.rand(1, 9)
    prediction, prob, _ = predict_flare(model, input_data, model_type, mean, std)
    assert prediction in ["No Flare Expected", "Flare Likely"]
    assert 0 <= prob <= 1