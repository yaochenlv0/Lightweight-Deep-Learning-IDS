import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from config import MLP_MODEL_PATH, RESULT_DIR, MLP_HIDDEN_LAYERS, MLP_EPOCHS, MLP_LR, BATCH_SIZE
from src.evaluate import calculate_metrics, plot_confusion_matrix, save_metrics
from src.visualization import plot_training_loss, plot_prediction_probability

class LightweightMLP(nn.Module):
    """轻量化多层感知机"""
    def __init__(self, input_dim, hidden_layers):
        super(LightweightMLP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_dl_model(X_train_scaled, y_train):
    """训练轻量化深度学习模型"""
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train_scaled.shape[1]
    model = LightweightMLP(input_dim=input_dim, hidden_layers=MLP_HIDDEN_LAYERS)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)

    loss_history = []

    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"[DL] Epoch {epoch + 1}/{MLP_EPOCHS}, Loss: {avg_loss:.4f}")

    return model, loss_history


def run_dl(X_train_scaled, X_test_scaled, y_train, y_test):
    """执行深度学习模型训练、评估、保存"""
    start_time = time.time()

    model,loss_history = train_dl_model(X_train_scaled, y_train)

    end_time = time.time()
    train_time = end_time - start_time

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
        y_pred = (probs >= 0.5).astype(int)

    plot_training_loss(loss_history, f"{RESULT_DIR}/mlp_loss_curve.png")
    plot_prediction_probability(probs, y_test, f"{RESULT_DIR}/mlp_probability_distribution.png")

    metrics = calculate_metrics(y_test, y_pred)
    metrics["train_time"] = train_time

    torch.save(model.state_dict(), MLP_MODEL_PATH)

    save_metrics(metrics, f"{RESULT_DIR}/dl_metrics.csv")
    plot_confusion_matrix(
        y_test,
        y_pred,
        f"{RESULT_DIR}/dl_confusion_matrix.png",
        title="Deep Learning Confusion Matrix"
    )

    return model, metrics