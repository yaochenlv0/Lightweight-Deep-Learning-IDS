import glob
import hashlib
import ipaddress
import os
import re

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    LABEL_COLUMN,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_PATH,
    DATASET_TYPE,
    KITSUNE_DATA_DIR,
    KITSUNE_ATTACK,
    KITSUNE_MAX_ROWS,
    KITSUNE_SAMPLE_MODE,
    KITSUNE_5TUPLE_DIR,
    KITSUNE_5TUPLE_FILE,
    CICIDS_DATA_DIR,
    CICIDS_FILES,
    CICIDS_MAX_ROWS,
    CICIDS_SAMPLE_MODE,
)


def _kitsune_file_candidates(data_dir):
    patterns = [
        os.path.join(data_dir, "**", "*_dataset.csv"),
        os.path.join(data_dir, "**", "*_dataset.csv.gz"),
        os.path.join(data_dir, "**", "*dataset*.csv"),
        os.path.join(data_dir, "**", "*dataset*.csv.gz"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def find_kitsune_pair(data_dir=KITSUNE_DATA_DIR, attack_name=KITSUNE_ATTACK):
    """查找 UCI Kitsune 的特征文件和标签文件。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"未找到 Kitsune 数据目录：{data_dir}")

    dataset_files = _kitsune_file_candidates(data_dir)
    if attack_name:
        keyword = attack_name.lower()
        dataset_files = [path for path in dataset_files if keyword in path.lower()]

    for dataset_path in dataset_files:
        label_candidates = []
        label_candidates.append(re.sub(r"_dataset(\.csv(?:\.gz)?)$", r"_labels\1", dataset_path, flags=re.IGNORECASE))
        current_dir = os.path.dirname(dataset_path)
        label_candidates.extend(sorted(glob.glob(os.path.join(current_dir, "*_labels.csv*"))))
        label_candidates.extend(sorted(glob.glob(os.path.join(current_dir, "*labels*.csv*"))))
        for label_path in label_candidates:
            if os.path.exists(label_path):
                return dataset_path, label_path

    hint = f"，攻击关键字：{attack_name}" if attack_name else ""
    raise FileNotFoundError(f"未在 {data_dir} 中找到 Kitsune 的 *_dataset.csv(.gz) 与 *_labels.csv(.gz) 配对文件{hint}")


def _read_kitsune_csv(path, nrows=None):
    return pd.read_csv(path, header=None, nrows=nrows, compression="infer", low_memory=False)


def load_kitsune_data(data_dir=KITSUNE_DATA_DIR, attack_name=KITSUNE_ATTACK, max_rows=KITSUNE_MAX_ROWS):
    """读取 UCI Kitsune 数据集，并合并特征文件与标签文件。"""
    dataset_path, label_path = find_kitsune_pair(data_dir, attack_name)
    print(f"使用 UCI Kitsune 特征文件：{dataset_path}")
    print(f"使用 UCI Kitsune 标签文件：{label_path}")

    nrows = max_rows if max_rows and max_rows > 0 else None
    features = _read_kitsune_csv(dataset_path, nrows=nrows)
    labels = _read_kitsune_csv(label_path, nrows=nrows)

    features = features.apply(pd.to_numeric, errors="coerce")
    features.columns = [f"kitsune_f_{idx:03d}" for idx in range(features.shape[1])]
    label_series = pd.to_numeric(labels.iloc[:, 0], errors="coerce").rename(LABEL_COLUMN)

    df = pd.concat([features, label_series], axis=1)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df = df[df[LABEL_COLUMN].isin([0, 1])]
    print("Kitsune 数据形状：", df.shape)
    return df


def find_kitsune_5tuple_files(data_dir=KITSUNE_5TUPLE_DIR, file_keyword=KITSUNE_5TUPLE_FILE):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"未找到 Kitsune 5-Tuple 数据目录：{data_dir}")
    files = [
        path for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        if os.path.getsize(path) > 0
    ]
    if file_keyword:
        keyword = file_keyword.lower()
        files = [path for path in files if keyword in os.path.basename(path).lower()]
    if not files:
        hint = f"，文件关键字：{file_keyword}" if file_keyword else ""
        raise FileNotFoundError(f"未在 {data_dir} 中找到 Kitsune 5-Tuple CSV 文件{hint}")
    return files


def find_kitsune_5tuple_file(data_dir=KITSUNE_5TUPLE_DIR, file_keyword=KITSUNE_5TUPLE_FILE):
    return min(find_kitsune_5tuple_files(data_dir, file_keyword), key=os.path.getsize)


def sample_with_natural_ratio(df, max_rows):
    if not max_rows or max_rows <= 0 or len(df) <= max_rows:
        return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def sample_with_balanced_classes(df, max_rows):
    if not max_rows or max_rows <= 0 or LABEL_COLUMN not in df.columns or len(df) <= max_rows:
        return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    grouped = []
    per_class = max(1, max_rows // max(df[LABEL_COLUMN].nunique(), 1))
    for _, part in df.groupby(LABEL_COLUMN, group_keys=False):
        grouped.append(part.sample(n=min(len(part), per_class), random_state=RANDOM_STATE))
    sampled = pd.concat(grouped, ignore_index=True)
    if len(sampled) < max_rows:
        remaining = df.drop(sampled.index.intersection(df.index), errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(n=min(len(remaining), max_rows - len(sampled)), random_state=RANDOM_STATE)],
                ignore_index=True,
            )
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_kitsune_5tuple_data(data_dir=KITSUNE_5TUPLE_DIR, file_keyword=KITSUNE_5TUPLE_FILE, max_rows=KITSUNE_MAX_ROWS, sample_mode=KITSUNE_SAMPLE_MODE):
    """读取 Kaggle 上的 Kitsune 5-Tuple 派生数据。"""
    files = find_kitsune_5tuple_files(data_dir, file_keyword)
    print("使用 Kitsune 5-Tuple 文件：")
    for path in files:
        print(f" - {path}")

    frames = []
    for path in files:
        part = pd.read_csv(path, low_memory=False)
        part["attack_source_file"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.astype(str).str.strip()
    label_cols = [col for col in df.columns if col.lower() == "label"]
    if label_cols and label_cols[0] != LABEL_COLUMN:
        df = df.rename(columns={label_cols[0]: LABEL_COLUMN})
    if sample_mode in {"balanced", "balance", "class_balanced"}:
        df = sample_with_balanced_classes(df, max_rows)
        print("Kitsune 5-Tuple 抽样方式：类别均衡抽样")
    else:
        df = sample_with_natural_ratio(df, max_rows)
        print("Kitsune 5-Tuple 抽样方式：保留原始比例随机抽样")
    print("Kitsune 5-Tuple 数据形状：", df.shape)
    return df


def find_cicids_files(data_dir=CICIDS_DATA_DIR, file_filter=CICIDS_FILES):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"未找到 CICIDS2017 数据目录：{data_dir}")
    files = [
        path for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        if os.path.getsize(path) > 0
    ]
    if file_filter:
        filters = [item.strip().lower() for item in file_filter.split(",") if item.strip()]
        files = [
            path for path in files
            if any(keyword in os.path.basename(path).lower() or keyword in path.lower() for keyword in filters)
        ]
    if not files:
        hint = f"，筛选条件：{file_filter}" if file_filter else ""
        raise FileNotFoundError(f"未在 {data_dir} 中找到 CICIDS2017 CSV 文件{hint}")
    return files


def _read_csv_sample(path, target_rows=None):
    if not target_rows or target_rows <= 0:
        return pd.read_csv(path, low_memory=False)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
        chunk.columns = chunk.columns.astype(str).str.strip()
        chunk["_sample_key"] = rng.random(len(chunk))
        if sampled.empty:
            sampled = chunk
        else:
            sampled = pd.concat([sampled, chunk], ignore_index=True)
        if len(sampled) > target_rows:
            sampled = sampled.nsmallest(target_rows, "_sample_key").reset_index(drop=True)
    if sampled.empty:
        return pd.DataFrame()
    return sampled.drop(columns=["_sample_key"], errors="ignore").reset_index(drop=True)


def load_cicids_data(data_dir=CICIDS_DATA_DIR, file_filter=CICIDS_FILES, max_rows=CICIDS_MAX_ROWS, sample_mode=CICIDS_SAMPLE_MODE):
    """读取 CICIDS2017 数据，作为与 Kitsune 分离的公开数据集实验。"""
    files = find_cicids_files(data_dir, file_filter)
    print("使用 CICIDS2017 文件：")
    for path in files:
        print(f" - {path}")

    total_size = sum(os.path.getsize(path) for path in files) or 1
    frames = []
    loaded_rows = 0
    for idx, path in enumerate(files):
        if max_rows and max_rows > 0:
            quota = max(1, int(max_rows * os.path.getsize(path) / total_size))
            if idx == len(files) - 1:
                quota = max(1, max_rows - loaded_rows)
        else:
            quota = None
        part = _read_csv_sample(path, quota)
        if part.empty:
            continue
        part.columns = part.columns.astype(str).str.strip()
        label_cols = [col for col in part.columns if col.lower() == "label"]
        if not label_cols:
            continue
        if label_cols[0] != LABEL_COLUMN:
            part = part.rename(columns={label_cols[0]: LABEL_COLUMN})
        raw_label = part[LABEL_COLUMN].astype(str).str.strip()
        part["attack_label"] = raw_label
        part["attack_source_file"] = os.path.splitext(os.path.basename(path))[0]
        part[LABEL_COLUMN] = raw_label.str.upper().apply(lambda value: 0 if value == "BENIGN" else 1)
        frames.append(part)
        loaded_rows += len(part)

    if not frames:
        raise ValueError("CICIDS2017 数据文件中没有可用的 Label 列。")

    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    if sample_mode in {"balanced", "balance", "class_balanced"}:
        df = sample_with_balanced_classes(df, max_rows)
        print("CICIDS2017 抽样方式：二分类均衡抽样")
    else:
        df = sample_with_natural_ratio(df, max_rows)
        print("CICIDS2017 抽样方式：保留原始比例随机抽样")
    print("CICIDS2017 数据形状：", df.shape)
    print("CICIDS2017 标签分布：")
    print(df[LABEL_COLUMN].value_counts())
    return df


def load_data(file_path):
    """读取原始数据集"""
    if DATASET_TYPE in {"kitsune", "uci_kitsune"}:
        return load_kitsune_data()
    if DATASET_TYPE in {"kitsune_5tuple", "kitsune5tuple"}:
        return load_kitsune_5tuple_data()
    if DATASET_TYPE in {"cicids", "cicids2017"}:
        return load_cicids_data()

    if DATASET_TYPE == "auto":
        try:
            find_kitsune_pair()
            return load_kitsune_data()
        except FileNotFoundError:
            pass
        try:
            find_kitsune_5tuple_file()
            return load_kitsune_5tuple_data()
        except FileNotFoundError:
            pass
        try:
            find_cicids_files()
            return load_cicids_data()
        except FileNotFoundError:
            pass

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


def encode_feature_column(series):
    """将类别特征稳定转换为数值，避免训练和页面检测时编码不一致。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype(str).str.strip()

    def encode_one(value):
        if value in {"", "nan", "None"}:
            return 0.0
        try:
            return float(int(ipaddress.ip_address(value)))
        except ValueError:
            digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
            return float(int(digest, 16))

    return text.map(encode_one)


def process_label(df):
    """标签处理：BENIGN -> 0，其余攻击类型 -> 1"""
    label_cols = [col for col in df.columns if col.lower() == "label"]
    if LABEL_COLUMN not in df.columns and label_cols:
        df = df.rename(columns={label_cols[0]: LABEL_COLUMN})
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
    helper_columns = [col for col in ["attack_source_file", "attack_label"] if col in df.columns]
    X = df.drop([LABEL_COLUMN] + helper_columns, axis=1)
    for col in X.columns:
        X[col] = encode_feature_column(X[col])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
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
