
import numpy as np

def augment_data(X, y, num_augs=10, noise_std=0.1):
    X_aug = [X]
    y_aug = [y]
    for _ in range(num_augs):
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std