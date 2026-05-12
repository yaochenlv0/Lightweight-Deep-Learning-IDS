import os
import matplotlib.pyplot as plt
import numpy as np


def plot_average_model_comparison(rf_avg_metrics, mlp_avg_metrics, save_path):
    """
    根据平均指标生成模型性能对比图
    平均指标已经手算得出
    参数：
    rf_avg_metrics: dict
        随机森林平均指标，例如：
        {
            "accuracy": 0.999769,
            "precision": 0.999877,
            "recall": 0.946619,
            "f1": 0.969684
        }

    mlp_avg_metrics: dict
        轻量化 MLP 平均指标，例如：
        {
            "accuracy": 0.998480,
            "precision": 0.971349,
            "recall": 0.884991,
            "f1": 0.913161
        }

    save_path: str
        图片保存路径
    """
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    rf_values = [
        rf_avg_metrics["accuracy"],
        rf_avg_metrics["precision"],
        rf_avg_metrics["recall"],
        rf_avg_metrics["f1"]
    ]

    mlp_values = [
        mlp_avg_metrics["accuracy"],
        mlp_avg_metrics["precision"],
        mlp_avg_metrics["recall"],
        mlp_avg_metrics["f1"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, rf_values, width, label="RandomForest")
    plt.bar(x + width / 2, mlp_values, width, label="Lightweight MLP")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Average Performance Comparison")
    plt.legend()

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"平均指标模型性能对比图已保存：{save_path}")


if __name__ == "__main__":
    rf_avg_metrics = {
        "accuracy": 0.999769,
        "precision": 0.999877,
        "recall": 0.946619,
        "f1": 0.969684
    }

    mlp_avg_metrics = {
        "accuracy": 0.998480,
        "precision": 0.971349,
        "recall": 0.884991,
        "f1": 0.913161
    }

    plot_average_model_comparison(
        rf_avg_metrics,
        mlp_avg_metrics,
        "运行记录/平均指标模型性能对比图/average_model_comparison.png"
    )