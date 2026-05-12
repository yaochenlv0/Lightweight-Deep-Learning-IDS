import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import LABEL_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE, SCALER_PATH


def load_data(file_path):
    """读取原始数据集"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("当前列名：", df.columns.tolist())
    return df


def clean_data(df):
    """数据清洗：处理无穷值、去缺失值、删无关列"""
    print("清洗前数据形状：", df.shape)

    df = df.replace([np.inf, -np.inf], np.nan)

    print("各列缺失值数量：")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    df = df.dropna()

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

    print("清洗后数据形状：", df.shape)

    return df


def process_label(df):
    """标签处理：BENIGN -> 0，其余攻击类型 -> 1"""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"找不到标签列 {LABEL_COLUMN}，当前列名为：{df.columns.tolist()}")

    if df[LABEL_COLUMN].dtype == object:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.upper()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: 0 if x == "BENIGN" else 1)

    df = df[df[LABEL_COLUMN].isin([0, 1])]

    print("标签分布：")
    print(df[LABEL_COLUMN].value_counts())

    return df


def split_features_labels(df):
    """划分特征和标签"""
    X = df.drop(LABEL_COLUMN, axis=1)
    y = df[LABEL_COLUMN]
    return X, y


def standardize_data(X_train, X_test):
    """标准化训练集和测试集，并保存 scaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(file_path):
    """完整数据预处理流程"""
    df = load_data(file_path)
    df = clean_data(df)
    df = process_label(df)

    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler