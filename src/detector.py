import joblib
import torch
import pandas as pd

from config import RF_MODEL_PATH, SCALER_PATH, MLP_MODEL_PATH, MLP_HIDDEN_LAYERS
from src.train_dl import LightweightMLP


class IntrusionDetector:
    """入侵检测器：支持 RF 和 MLP 两种模型预测"""
    def __init__(self, input_dim):
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        self.mlp_model = LightweightMLP(input_dim=input_dim, hidden_layers=MLP_HIDDEN_LAYERS)
        self.mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=torch.device("cpu")))
        self.mlp_model.eval()

    def predict_rf(self, sample_df):
        """使用随机森林预测"""
        pred = self.rf_model.predict(sample_df)[0]
        return int(pred)

    def predict_mlp(self, sample_df):
        """使用轻量化 MLP 预测"""
        sample_scaled = self.scaler.transform(sample_df)
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(self.mlp_model(sample_tensor)).item()
            pred = 1 if prob >= 0.5 else 0

        return pred, prob