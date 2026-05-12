import os
import pandas as pd

from config import DATA_PATH, MODEL_DIR, RESULT_DIR
from src.utils import ensure_dir
from src.data_preprocessing import prepare_data
from src.train_baseline import run_baseline
from src.train_dl import run_dl
from src.evaluate import plot_comparison
from src.detector import IntrusionDetector
from src.visualization import plot_label_distribution, plot_feature_importance

def main():
    print("=" * 60)
    print("基于深度学习的轻量化入侵检测系统启动")
    print("=" * 60)

    # 创建模型和结果保存目录
    ensure_dir(MODEL_DIR)
    ensure_dir(RESULT_DIR)

    # 1. 数据预处理
    print("\n[1] 开始数据预处理...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(DATA_PATH)
    print("数据预处理完成")
    plot_label_distribution(y_train, f"{RESULT_DIR}/label_distribution.png")
    print("标签分布图已保存")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 2. 训练 baseline 模型
    print("\n[2] 开始训练基线模型 RandomForest...")
    baseline_model, baseline_metrics = run_baseline(X_train, X_test, y_train, y_test)
    print("Baseline 指标:")
    print(baseline_metrics)

    if "train_time" in baseline_metrics:
        print(f"Baseline 训练时间: {baseline_metrics['train_time']:.4f} 秒")

    plot_feature_importance(
        baseline_model,
        X_train.columns,
        f"{RESULT_DIR}/rf_feature_importance.png",
        top_n=15
    )

    print("特征重要性图已保存")
    # 3. 训练轻量化深度学习模型
    print("\n[3] 开始训练轻量化 MLP 模型...")
    dl_model, dl_metrics = run_dl(X_train_scaled, X_test_scaled, y_train, y_test)
    print("DL 指标:")
    print(dl_metrics)

    if "train_time" in dl_metrics:
        print(f"DL 训练时间: {dl_metrics['train_time']:.4f} 秒")

    # 4. 生成模型对比图
    print("\n[4] 生成模型对比图...")
    plot_comparison(
        baseline_metrics,
        dl_metrics,
        f"{RESULT_DIR}/comparison.png"
    )
    print("对比图已保存")

    # 5. 系统检测演示：从测试集取 1 条样本进行预测
    print("\n[5] 系统检测演示...")
    sample_df = X_test.iloc[[0]]
    true_label = y_test.iloc[0]

    detector = IntrusionDetector(input_dim=X_train.shape[1])

    rf_pred = detector.predict_rf(sample_df)
    mlp_pred, mlp_prob = detector.predict_mlp(sample_df)

    print("单样本检测结果：")
    print(f"真实标签: {true_label}")
    print(f"RandomForest 预测: {rf_pred}")
    print(f"MLP 预测: {mlp_pred}, 攻击概率: {mlp_prob:.4f}")

    print("\n系统运行完成！")
    print(f"模型保存在: {MODEL_DIR}")
    print(f"结果保存在: {RESULT_DIR}")


if __name__ == "__main__":
    main()