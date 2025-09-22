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
    # Add sleep efficiency
    df['sleep_efficiency'] = df['total_sleep'] / (df['total_sleep'] + df['wake'])
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