import numpy as np

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def augment_data(X, y, num_augs=5, noise_std=0.05):
    X_aug = [X]
    y_aug = [y]
    flare_idx = np.where(y == 1)[0]
    for _ in range(num_augs):
        for idx in flare_idx:
            noise = np.random.normal(0, noise_std, X[idx].shape)
            X_aug.append(X[idx] + noise)
            y_aug.append(y[idx])
    return np.vstack(X_aug), np.hstack(y_aug)