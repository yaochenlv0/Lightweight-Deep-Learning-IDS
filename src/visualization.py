# src/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_label_distribution(y, save_path):
    """绘制标签分布图"""
    counts = y.value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["Normal(0)", "Attack(1)"], counts.values)
    plt.title("Label Distribution")
    plt.ylabel("Count")

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_loss(loss_history, save_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """绘制随机森林特征重要性图"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_prediction_probability(probs, y_true, save_path):
    """绘制预测概率分布图"""
    probs = np.array(probs).flatten()
    y_true = np.array(y_true)

    normal_probs = probs[y_true == 0]
    attack_probs = probs[y_true == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_probs, bins=30, alpha=0.6, label="Normal(0)")
    plt.hist(attack_probs, bins=30, alpha=0.6, label="Attack(1)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()