import joblib
import time
from sklearn.ensemble import RandomForestClassifier

from config import RF_MODEL_PATH, RESULT_DIR
from src.evaluate import calculate_metrics, plot_confusion_matrix, save_metrics


def train_baseline_model(X_train, y_train):
    """训练随机森林基线模型"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def run_baseline(X_train, X_test, y_train, y_test):
    """执行 baseline 训练、评估、保存"""
    start_time = time.time()

    model = train_baseline_model(X_train, y_train)
    end_time = time.time()
    y_pred = model.predict(X_test)

    train_time = end_time - start_time

    metrics = calculate_metrics(y_test, y_pred)
    metrics["train_time"] = train_time

    joblib.dump(model, RF_MODEL_PATH)

    save_metrics(metrics, f"{RESULT_DIR}/baseline_metrics.csv")
    plot_confusion_matrix(
        y_test,
        y_pred,
        f"{RESULT_DIR}/baseline_confusion_matrix.png",
        title="Baseline Confusion Matrix"
    )

    return model, metrics