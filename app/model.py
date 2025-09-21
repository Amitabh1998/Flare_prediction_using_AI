import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score
from app.data import get_data_splits, FlareDataset
from app.utils import normalize, augment_data

class FlareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_all_models(df, features, model_path):
    X = df[features].values
    y = df['flare_next_day'].values

    train_idx, test_idx, X_train, X_test, y_train, y_test = get_data_splits(X, y)
    X_train_norm, mean, std = normalize(X_train)
    X_test_norm = (X_test - mean) / std
    X_train_aug, y_train_aug = augment_data(X_train_norm, y_train)

    # Train NN
    train_dataset = FlareDataset(X_train_aug, y_train_aug)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model_nn = FlareNet()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

    for epoch in range(200):
        model_nn.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_nn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    y_pred_nn = model_nn(torch.tensor(X_test_norm, dtype=torch.float32)).argmax(dim=1).numpy()
    acc_nn, recall_no_nn, recall_flare_nn = evaluate(y_test, y_pred_nn)

    # Train Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_rf.fit(X_train_norm, y_train)
    y_pred_rf = model_rf.predict(X_test_norm)
    acc_rf, recall_no_rf, recall_flare_rf = evaluate(y_test, y_pred_rf)

    # Train XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
    model_xgb = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=scale_pos_weight)
    model_xgb.fit(X_train_norm, y_train)
    y_pred_xgb = model_xgb.predict(X_test_norm)
    acc_xgb, recall_no_xgb, recall_flare_xgb = evaluate(y_test, y_pred_xgb)

    # Train Logistic Regression
    model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model_lr.fit(X_train_norm, y_train)
    y_pred_lr = model_lr.predict(X_test_norm)
    acc_lr, recall_no_lr, recall_flare_lr = evaluate(y_test, y_pred_lr)

    # Select best model based on average recall
    models = {
        'NN': (model_nn, acc_nn, (recall_no_nn + recall_flare_nn) / 2, mean, std),
        'RF': (model_rf, acc_rf, (recall_no_rf + recall_flare_rf) / 2, mean, std),
        'XGB': (model_xgb, acc_xgb, (recall_no_xgb + recall_flare_xgb) / 2, mean, std),
        'LR': (model_lr, acc_lr, (recall_no_lr + recall_flare_lr) / 2, mean, std)
    }

    best_model_type = max(models, key=lambda k: models[k][2])
    best_model, best_acc, best_avg_recall, mean, std = models[best_model_type]
    report = f"Best Model: {best_model_type}\nAccuracy: {best_acc*100:.2f}%\nRecall (No Flare): {recall_no_nn*100:.2f}%\nRecall (Flare): {recall_flare_nn*100:.2f}%"

    # Save best model
    if best_model_type == 'NN':
        torch.save(best_model.state_dict(), model_path)
    else:
        import joblib
        joblib.dump(best_model, model_path)

    return best_model, best_model_type, best_acc, report, mean, std

def load_model(model_path, model_type):
    if model_type == 'NN':
        model = FlareNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        import joblib
        model = joblib.load(model_path)
    return model

def evaluate(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    recall_no = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_flare = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    return acc, recall_no, recall_flare

def predict_flare(model, input_data, model_type, mean, std):
    input_norm = (input_data - mean) / std
    if model_type == 'NN':
        model.eval()
        input_tensor = torch.tensor(input_norm, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            prob_flare = probabilities[1].item()
            return "Flare Likely" if prediction == 1 else "No Flare Expected", prob_flare, output.numpy()
    else:
        input_norm = input_norm.reshape(1, -1)
        prediction = model.predict(input_norm)[0]
        probabilities = model.predict_proba(input_norm)[0]
        prob_flare = probabilities[1]
        return "Flare Likely" if prediction == 1 else "No Flare Expected", prob_flare, None