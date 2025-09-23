import pandas as pd
from app.data import load_data, get_data_splits

def test_load_data():
    df = load_data()
    assert df.shape[0] <= 252, f"Expected up to 252 rows, got {df.shape[0]}"
    assert df.shape[1] >= 15, f"Expected at least 15 columns, got {df.shape[1]}"
    expected_columns = ['day', 'total_sleep', 'deep_sleep', 'rem_sleep', 'wake', 'subject', 'pain', 'fatigue', 'mood', 'sleep_hours', 'treatment', 'flare_next_day', 'sleep_efficiency', 'pain_fatigue_interaction', 'mood_pain_interaction', 'pain_rolling_mean']
    assert all(col in df.columns for col in expected_columns), "Missing required columns"
    assert df['total_sleep'].ge(df['deep_sleep'] + df['rem_sleep']).all(), "Total sleep must be >= deep_sleep + rem_sleep"
    assert df['pain'].between(0, 10).all(), "Pain must be in [0, 10]"
    assert df['fatigue'].between(0, 10).all(), "Fatigue must be in [0, 10]"
    assert df['mood'].between(0, 10).all(), "Mood must be in [0, 10]"
    assert df['treatment'].isin([0, 1]).all(), "Treatment must be 0 or 1"
    assert df['flare_next_day'].isin([0, 1]).all(), "Flare_next_day must be 0 or 1"
    assert df['sleep_efficiency'].between(0, 1).all(), "Sleep efficiency must be in [0, 1]"
    assert df['pain_fatigue_interaction'].ge(0).all(), "Pain-fatigue interaction must be >= 0"
    assert df['mood_pain_interaction'].ge(0).all(), "Mood-pain interaction must be >= 0"
    assert df['pain_rolling_mean'].between(0, 10).all(), "Pain rolling mean must be in [0, 10]"
    assert df['total_sleep'].le(24).all(), "Total sleep must be <= 24"
    assert df['wake'].le(24).all(), "Wake time must be <= 24"
    assert df['sleep_hours'].le(2).all(), "Sleep hours must be <= 2"
    assert df['sleep_hours'].ge(0.1).all(), "Sleep hours must be >= 0.1"
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

def test_data_splits():
    df = load_data()
    features = ['pain', 'fatigue', 'sleep_hours', 'sleep_efficiency', 'pain_fatigue_interaction', 'mood_pain_interaction', 'pain_rolling_mean']
    X = df[features].values
    y = df['flare_next_day'].values
    train_idx, test_idx, X_train, X_test, y_train, y_test = get_data_splits(X, y)
    assert len(train_idx) == int(0.8 * len(X)), "Train split size incorrect"
    assert len(test_idx) == len(X) - len(train_idx), "Test split size incorrect"
    assert X_train.shape[0] == len(train_idx), "X_train size mismatch"
    assert X_test.shape[0] == len(test_idx), "X_test size mismatch"
    assert y_train.shape[0] == len(train_idx), "y_train size mismatch"
    assert y_test.shape[0] == len(test_idx), "y_test size mismatch"
    assert X_train.shape[1] == 7, f"Expected 7 features, got {X_train.shape[1]}"