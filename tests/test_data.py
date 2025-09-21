import pandas as pd
import pytest
import numpy as np
from app.data import load_data, get_data_splits

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame), "Data should be a pandas DataFrame"
    assert len(df) == 120, f"Expected 108 rows, got {len(df)}"
    expected_columns = {'day', 'total_sleep', 'deep_sleep', 'rem_sleep', 'wake', 'subject', 
                        'pain', 'fatigue', 'mood', 'sleep_hours', 'treatment', 'flare_next_day'}
    assert set(df.columns) == expected_columns, f"Unexpected columns: {set(df.columns).symmetric_difference(expected_columns)}"
    assert df['flare_next_day'].isin([0, 1]).all(), "flare_next_day should only contain 0 or 1"
    assert df['pain'].between(0, 10).all(), "pain should be between 0 and 10"
    assert df['fatigue'].between(0, 10).all(), "fatigue should be between 0 and 10"
    assert df['mood'].between(0, 10).all(), "mood should be between 0 and 10"

def test_data_splits():
    df = load_data()
    X = df[['total_sleep', 'deep_sleep', 'rem_sleep', 'wake', 'pain', 'fatigue', 'mood', 'sleep_hours', 'treatment']].values
    y = df['flare_next_day'].values
    train_idx, test_idx, X_train, X_test, y_train, y_test = get_data_splits(X, y)
    assert len(train_idx) == int(0.8 * len(X)), f"Expected {int(0.8 * len(X))} training samples, got {len(train_idx)}"
    assert len(test_idx) == len(X) - len(train_idx), f"Expected {len(X) - len(train_idx)} test samples, got {len(test_idx)}"
    assert X_train.shape[0] == len(y_train), f"Mismatch in X_train ({X_train.shape[0]}) and y_train ({len(y_train)})"
    assert X_test.shape[0] == len(y_test), f"Mismatch in X_test ({X_test.shape[0]}) and y_test ({len(y_test)})"
    assert X_train.shape[1] == 9, f"Expected 9 features in X_train, got {X_train.shape[1]}"