import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import RESULT_DIR
from src.utils import save_dict_to_csv


def calculate_metrics(y_true, y_pred):
    """计算分类指标"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """绘制并保存混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, save_path):
    """保存指标到 CSV"""
    save_dict_to_csv(metrics, save_path)


def plot_comparison(baseline_metrics, dl_metrics, save_path):
    """绘制 baseline 与深度学习模型指标对比图"""
    metric_names = ["accuracy", "precision", "recall", "f1"]
    baseline_values = [baseline_metrics[m] for m in metric_names]
    dl_values = [dl_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, baseline_values, width, label="Baseline")
    plt.bar(x + width / 2, dl_values, width, label="Lightweight MLP")

    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Baseline vs Lightweight Deep Learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(rf_metrics, mlp_metrics, save_path):
    """
    rf_metrics: dict -> {"accuracy":xx, "precision":xx, "recall":xx, "f1":xx}
    mlp_metrics: dict -> 同上
    """

    labels = ["Accuracy", "Precision", "Recall", "F1"]

    rf_values = [
        rf_metrics["accuracy"],
        rf_metrics["precision"],
        rf_metrics["recall"],
        rf_metrics["f1"]
    ]

    mlp_values = [
        mlp_metrics["accuracy"],
        mlp_metrics["precision"],
        mlp_metrics["recall"],
        mlp_metrics["f1"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()

    plt.bar(x - width/2, rf_values, width, label="RandomForest")
    plt.bar(x + width/2, mlp_values, width, label="Lightweight MLP")

    plt.xticks(x, labels)
    plt.ylim(0, 1)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")

    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("模型对比图已保存：", save_path)