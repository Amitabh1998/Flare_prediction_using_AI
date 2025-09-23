# app/data.py
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class FlareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    df = pd.read_csv(data_path)
    # Clean data
    df = df[df['total_sleep'] >= df['deep_sleep'] + df['rem_sleep']]
    df['pain'] = df['pain'].clip(0, 10)
    df['fatigue'] = df['fatigue'].clip(0, 10)
    df['mood'] = df['mood'].clip(0, 10)
    df['treatment'] = df['treatment'].clip(0, 1)
    df['flare_next_day'] = df['flare_next_day'].clip(0, 1)
    df['total_sleep'] = df['total_sleep'].clip(0, 24)
    df['wake'] = df['wake'].clip(0, 24)
    df['sleep_hours'] = df['sleep_hours'].clip(0.1, 2)  # Address outlier (min=0.002)
    # Add features
    df['sleep_efficiency'] = df['total_sleep'] / (df['total_sleep'] + df['wake']).replace(0, 1e-10)
    df['pain_fatigue_interaction'] = df['pain'] * df['fatigue']
    df['mood_pain_interaction'] = df['mood'] * df['pain']
    # Add rolling mean of pain per subject (3-day window)
    df['pain_rolling_mean'] = df.groupby('subject')['pain'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['pain_rolling_mean'] = df['pain_rolling_mean'].fillna(df['pain'])  # Fill NaN for first entries
    return df

def get_data_splits(X, y):
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return train_idx, test_idx, X_train, X_test, y_train, y_test