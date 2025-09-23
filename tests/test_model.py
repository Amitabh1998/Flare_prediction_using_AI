import numpy as np
from app.model import train_all_models, predict_flare
from app.data import load_data

def test_model_inference():
    df = load_data()
    features = ['pain', 'fatigue', 'sleep_hours', 'sleep_efficiency', 'pain_fatigue_interaction', 'mood_pain_interaction', 'pain_rolling_mean']
    model, model_type, _, _, mean, std = train_all_models(df, features, 'models/test_model.pt')
    input_data = np.random.rand(1, 7)  # 7 features
    prediction, prob, _ = predict_flare(model, input_data, model_type, mean, std)
    assert prediction in ["No Flare Expected", "Flare Likely"]
    assert 0 <= prob <= 1