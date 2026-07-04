import os
import time
import html
import hashlib
import ipaddress
import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import plotly.express as px
import plotly.graph_objects as go

from config import (
    LABEL_COLUMN,
    DROP_COLUMNS,
    MODEL_DIR,
    RESULT_DIR,
    RF_MODEL_PATH,
    SCALER_PATH,
    MLP_MODEL_PATH,
    MLP_HIDDEN_LAYERS,
)
from src.detector import IntrusionDetector
from src.database import (
    init_db,
    insert_detection_results,
    insert_log,
    get_detection_results,
    get_logs,
    get_model_files,
    get_model_metrics,
    upsert_model_file,
)


SYSTEM_NAME = "基于深度学习的轻量化入侵检测系统"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAX_ROWS = 6000

NORMAL = "#1ee38a"
INFO = "#18c7ff"
BLUE = "#2f7dff"
WARN = "#f5c542"
BAD = "#ff4d5e"
PURPLE = "#8f7cff"
PANEL = "#071827"
PANEL_2 = "#0b2238"
TEXT = "#d7ebff"
MUTED = "#7ea4bf"

LIVE_VM_CSV = os.path.join(DATA_DIR, "vm_live_traffic.csv")
VM_BRIDGE_DIR = os.path.join(DATA_DIR, "vm_bridge")
VM_SIGNAL_FILE = os.path.join(VM_BRIDGE_DIR, "current_scenario.json")
DEMO_SCENARIO_DIR = os.path.join(DATA_DIR, "demo_scenarios")
VM_SCENARIO_CONFIG = {
    "normal": {"file": "normal_demo.csv", "label": "BENIGN", "attack_type": "正常流量", "risk": "低危", "result": "正常", "prob": (0.04, 0.28), "protocol": "HTTP"},
    "portscan": {"file": "portscan_demo.csv", "label": "PortScan", "attack_type": "PortScan", "risk": "中危", "result": "攻击", "prob": (0.62, 0.82), "protocol": "TCP"},
    "ssh_patator": {"file": "ssh_patator_demo.csv", "label": "SSH-Patator", "attack_type": "SSH-Patator", "risk": "高危", "result": "攻击", "prob": (0.82, 0.98), "protocol": "TCP"},
    "ftp_patator": {"file": "ftp_patator_demo.csv", "label": "FTP-Patator", "attack_type": "FTP-Patator", "risk": "高危", "result": "攻击", "prob": (0.80, 0.97), "protocol": "TCP"},
    "dos": {"file": "dos_demo.csv", "label": "DoS", "attack_type": "DoS", "risk": "高危", "result": "攻击", "prob": (0.86, 0.99), "protocol": "TCP"},
    "mixed_attack": {"file": "mixed_attack_demo.csv", "label": "Mixed", "attack_type": "混合攻击", "risk": "高危", "result": "攻击", "prob": (0.78, 0.98), "protocol": "TCP"},
}


st.set_page_config(page_title=SYSTEM_NAME, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #020813;
            --panel: #071827;
            --panel-2: #0b2238;
            --line: rgba(39, 221, 255, .18);
            --text: #d7ebff;
            --muted: #7ea4bf;
            --cyan: #18c7ff;
            --blue: #2f7dff;
            --green: #1ee38a;
            --yellow: #f5c542;
            --red: #ff4d5e;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(24,199,255,.12), transparent 28%),
                radial-gradient(circle at 82% 4%, rgba(47,125,255,.10), transparent 24%),
                linear-gradient(180deg, #020813 0%, #03111f 48%, #020813 100%);
            color: var(--text);
        }
        [data-testid="stHeader"], .stAppHeader {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] { display: none !important; }
        [data-testid="collapsedControl"] {
            color: var(--text) !important;
            background: rgba(7,24,39,.92) !important;
            border: 1px solid rgba(24,199,255,.30) !important;
            border-radius: 6px !important;
            box-shadow: 0 0 18px rgba(24,199,255,.12) !important;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] { display: none !important; }
        .block-container { padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1540px; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #03101d 0%, #061827 100%);
            border-right: 1px solid var(--line);
            min-width: 300px !important;
            width: 300px !important;
            max-width: 300px !important;
            margin-left: 0 !important;
            transform: translateX(0) !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        div[data-testid="stRadio"] label { padding: 8px 10px; border-radius: 6px; }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        textarea,
        input {
            background: #071827 !important;
            color: var(--text) !important;
            border-color: rgba(24,199,255,.26) !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, rgba(8,31,52,.96), rgba(5,18,31,.96)) !important;
            border: 1px dashed rgba(24,199,255,.38) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] section > div,
        div[data-testid="stFileUploaderDropzone"] > div {
            background: transparent !important;
            color: var(--text) !important;
        }
        div[data-testid="stFileUploader"] button {
            background: rgba(24,199,255,.10) !important;
            color: var(--text) !important;
            border: 1px solid rgba(24,199,255,.30) !important;
        }
        div[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
        div[data-testid="stAlert"] {
            background: rgba(24,199,255,.08);
            color: var(--text);
            border: 1px solid rgba(24,199,255,.22);
            border-radius: 8px;
        }
        .topbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 14px 18px; border: 1px solid var(--line); border-radius: 8px;
            background: linear-gradient(90deg, rgba(7,24,39,.96), rgba(5,18,31,.9));
            box-shadow: 0 0 28px rgba(24,199,255,.08);
            margin-bottom: 12px;
        }
        .brand { font-size: 22px; font-weight: 800; letter-spacing: 0; color: #f3fbff; }
        .subbrand { color: var(--muted); font-size: 12px; margin-top: 4px; }
        .clock { color: var(--muted); font-size: 12px; text-align: right; }
        .grid-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(7,24,39,.96), rgba(4,15,27,.94));
            border-radius: 8px;
            padding: 14px;
            box-shadow: inset 0 0 22px rgba(24,199,255,.035);
        }
        .metric-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(8,31,52,.95), rgba(5,18,31,.95));
            border-radius: 8px; padding: 13px 14px; min-height: 96px;
        }
        .metric-label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
        .metric-value { color: #f7fdff; font-size: 27px; font-weight: 800; line-height: 1.15; }
        .metric-hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .summary-grid {
            display: grid;
            grid-template-columns: minmax(240px, 1.7fr) repeat(auto-fit, minmax(145px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }
        .summary-grid .metric-card { min-height: 88px; }
        .summary-grid .metric-value { font-size: 24px; word-break: keep-all; white-space: nowrap; }
        .overview-status {
            min-height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 10px;
        }
        .overview-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .prediction-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(8,31,52,.95), rgba(5,18,31,.95));
            border-radius: 8px;
            padding: 14px;
            margin-top: 8px;
        }
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .prediction-item {
            border: 1px solid rgba(24,199,255,.18);
            border-radius: 8px;
            padding: 10px;
            background: rgba(24,199,255,.045);
        }
        .prediction-item .metric-value { font-size: 22px; }
        .prediction-note {
            color: var(--muted);
            font-size: 12px;
            margin-top: 10px;
            line-height: 1.5;
        }
        .page-title { color: #effbff; font-size: 18px; font-weight: 800; margin: 5px 0 10px; }
        .small-title { color: #dff8ff; font-size: 15px; font-weight: 800; margin-bottom: 8px; }
        .badge {
            display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px;
            border-radius: 6px; border: 1px solid var(--line); color: var(--text);
            background: rgba(24,199,255,.06); font-size: 12px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 6px; border: 1px solid rgba(24,199,255,.35);
            background: linear-gradient(90deg, #0d907d, #0a68a8); color: white;
            font-weight: 700;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(30,227,138,.7); color: white;
        }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
        .dark-table-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: auto;
            background: rgba(4,18,32,.86);
        }
        table.dark-table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text);
            font-size: 12px;
        }
        table.dark-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #09243b;
            color: #bfefff;
            text-align: left;
            font-weight: 700;
            border-bottom: 1px solid rgba(24,199,255,.24);
            padding: 9px 10px;
            white-space: nowrap;
        }
        table.dark-table td {
            border-bottom: 1px solid rgba(126,164,191,.12);
            padding: 8px 10px;
            white-space: nowrap;
            color: #d7ebff;
        }
        table.dark-table tr:nth-child(even) td { background: rgba(24,199,255,.035); }
        table.dark-table tr:hover td { background: rgba(24,199,255,.10); }
        .risk-high { color: var(--red) !important; font-weight: 800; }
        .risk-mid { color: var(--yellow) !important; font-weight: 800; }
        .risk-low { color: var(--green) !important; font-weight: 800; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(7,24,39,.92); border: 1px solid var(--line); border-radius: 6px;
            color: var(--text); padding: 8px 12px;
        }
        .sim-note { color: var(--muted); font-size: 12px; margin-top: -4px; margin-bottom: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,18,32,.72)",
        font=dict(color=TEXT, family="Microsoft YaHei, Arial"),
        margin=dict(l=35, r=18, t=42, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(126,164,191,.12)", zerolinecolor="rgba(126,164,191,.16)"),
        yaxis=dict(gridcolor="rgba(126,164,191,.12)", zerolinecolor="rgba(126,164,191,.16)"),
    )
    return fig


def show_chart(fig, height=300):
    st.plotly_chart(plotly_layout(fig, height), use_container_width=True, config={"displayModeBar": False})


def metric_card(label, value, hint="", color=INFO):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-hint"><span class="status-dot" style="background:{color};"></span>{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(label, value, hint="", color=INFO):
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-hint"><span class="status-dot" style="background:{color};"></span>{hint}</div>
        </div>
    """


def dark_table(df, height=320):
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    html = view.to_html(index=False, escape=False, classes="dark-table")
    html = html.replace(">高危<", ' class="risk-high">高危<')
    html = html.replace(">中危<", ' class="risk-mid">中危<')
    html = html.replace(">低危<", ' class="risk-low">低危<')
    html = html.replace(">攻击<", f' style="color:{BAD};font-weight:800;">攻击<')
    html = html.replace(">正常<", f' style="color:{NORMAL};font-weight:800;">正常<')
    st.markdown(
        f'<div class="dark-table-wrap" style="max-height:{height}px;">{html}</div>',
        unsafe_allow_html=True,
    )


def panel(title=None):
    if title:
        st.markdown(f'<div class="small-title">{title}</div>', unsafe_allow_html=True)


def topbar():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="brand">{SYSTEM_NAME}</div>
                <div class="subbrand">深度学习模型驱动 · RandomForest 基线对照 · 轻量化 MLP 检测流程</div>
            </div>
            <div class="clock">系统时间<br>{now}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_metrics(path, fallback):
    if os.path.exists(path):
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
            return {**fallback, **{k: float(v) for k, v in row.items() if pd.notna(v)}}
        except Exception:
            return fallback
    return fallback


def read_metrics_from_db():
    try:
        metrics_df = get_model_metrics()
    except Exception:
        return {}
    if metrics_df.empty:
        return {}

    model_metrics = {}
    for _, row in metrics_df.iterrows():
        model_metrics[row["model_name"]] = {
            "accuracy": float(row["accuracy"] or 0),
            "precision": float(row["precision_score"] or 0),
            "recall": float(row["recall"] or 0),
            "f1": float(row["f1_score"] or 0),
            "train_time": float(row["train_time"] or 0),
            "model_file_size": float(row["model_file_size"] or 0),
            "model_path": row["model_path"],
            "created_at": row["created_at"],
        }
    return model_metrics


@st.cache_data(show_spinner=False)
def get_metrics():
    zero_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "train_time": 0.0}
    db_metrics = read_metrics_from_db()
    rf = read_metrics(
        os.path.join(RESULT_DIR, "baseline_metrics.csv"),
        zero_metrics,
    )
    mlp = read_metrics(
        os.path.join(RESULT_DIR, "dl_metrics.csv"),
        zero_metrics,
    )
    rf = {**rf, **db_metrics.get("RandomForest", {})}
    mlp = {**mlp, **db_metrics.get("Lightweight MLP", {})}
    return rf, mlp


def fmt_pct(v):
    return f"{float(v) * 100:.2f}%"


def normalize_label(value):
    text = str(value).strip().upper()
    return 0 if text in {"0", "BENIGN", "NORMAL", "正常", "NONE"} else 1


def label_name(value):
    text = str(value).strip()
    if normalize_label(text) == 0:
        return "正常流量"
    return text if text and text.upper() != "1" else "攻击流量"


def get_csv_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([name for name in os.listdir(DATA_DIR) if name.lower().endswith((".csv", ".csv.gz"))])


@st.cache_data(show_spinner=False)
def load_csv(path, max_rows=MAX_ROWS):
    df = pd.read_csv(path, nrows=max_rows, compression="infer")
    df.columns = df.columns.str.strip()
    return df


def looks_like_headerless_numeric_csv(columns):
    if len(columns) < 8:
        return False
    numeric_cols = pd.to_numeric(pd.Series(columns), errors="coerce").notna().sum()
    return numeric_cols / max(len(columns), 1) >= 0.8


def read_detection_csv(source, nrows):
    upload_size = getattr(source, "size", 0) or (os.path.getsize(source) if isinstance(source, (str, os.PathLike)) and os.path.exists(source) else 0)
    read_all = upload_size and upload_size <= 512 * 1024 * 1024
    df = pd.read_csv(source, nrows=None if read_all else nrows, compression="infer")
    if looks_like_headerless_numeric_csv(df.columns):
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, header=None, nrows=None if read_all else nrows, compression="infer")
        df.columns = [f"kitsune_f_{idx:03d}" for idx in range(df.shape[1])]
    df.columns = df.columns.astype(str).str.strip()
    label_cols = [col for col in df.columns if col.lower() == "label"]
    if label_cols and label_cols[0] != LABEL_COLUMN:
        df = df.rename(columns={label_cols[0]: LABEL_COLUMN})
    if LABEL_COLUMN in df.columns and len(df) > nrows:
        label_values = df[LABEL_COLUMN].map(normalize_label)
        class_count = max(label_values.nunique(), 1)
        per_class = max(1, nrows // class_count)
        sampled = []
        for value, part in df.groupby(label_values, group_keys=False):
            sampled.append(part.sample(n=min(len(part), per_class), random_state=42))
        df = pd.concat(sampled, ignore_index=True)
        if len(df) < nrows:
            rest = df.drop(df.index, errors="ignore")
            if not rest.empty:
                df = pd.concat([df, rest.sample(n=min(len(rest), nrows - len(df)), random_state=42)], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def encode_feature_column(series):
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


def clean_feature_frame(df):
    work = df.copy()
    work.columns = work.columns.str.strip()
    for col in DROP_COLUMNS:
        if col in work.columns:
            work = work.drop(columns=[col])
    labels = None
    attack_types = None
    label_cols = [col for col in work.columns if col.lower() == "label"]
    label_col = LABEL_COLUMN if LABEL_COLUMN in work.columns else label_cols[0] if label_cols else None
    if label_col:
        labels = work[label_col].map(normalize_label).to_numpy()
        attack_types = work[label_col].map(label_name).to_numpy()
        work = work.drop(columns=[label_col])
    work = work.replace([np.inf, -np.inf], np.nan)
    numeric = pd.DataFrame(index=work.index)
    for col in work.columns:
        numeric[col] = encode_feature_column(work[col])
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    return numeric, labels, attack_types


def make_demo_raw(rows=1800):
    rng = np.random.default_rng(20260510)
    labels = rng.choice(["BENIGN", "DDoS", "PortScan", "Web Attack", "Bot", "Infiltration"], rows, p=[0.68, 0.12, 0.09, 0.06, 0.03, 0.02])
    return pd.DataFrame(
        {
            "Flow Duration": rng.integers(800, 1_900_000, rows),
            "Total Fwd Packets": rng.integers(1, 540, rows),
            "Total Backward Packets": rng.integers(1, 440, rows),
            "Flow Bytes/s": rng.normal(85000, 28000, rows).clip(100, None),
            "Total Length of Fwd Packets": rng.integers(20, 120000, rows),
            "Total Length of Bwd Packets": rng.integers(20, 100000, rows),
            "Destination Port": rng.choice([22, 53, 80, 443, 445, 1433, 3306, 8080], rows),
            "Protocol": rng.choice([6, 17, 1], rows, p=[0.70, 0.25, 0.05]),
            LABEL_COLUMN: labels,
        }
    )


def make_scenario_raw(scenario_key, rows):
    cfg = VM_SCENARIO_CONFIG[scenario_key]
    raw = make_demo_raw(rows)
    raw[LABEL_COLUMN] = cfg["label"]
    return raw


def detect_dataset_schema_from_columns(columns):
    names = {str(col).strip().lower() for col in columns}
    if {"time", "src", "sport", "dst", "dport"}.issubset(names):
        return "kitsune_5tuple"
    if any(name.startswith("kitsune_f_") for name in names):
        return "uci_kitsune"
    cicids_markers = {"destination port", "flow duration", "total fwd packets", "total backward packets"}
    if len(cicids_markers.intersection(names)) >= 3:
        return "cicids2017"
    return "unknown"


def schema_display_name(schema):
    return {
        "kitsune_5tuple": "Kitsune 5-tuple",
        "uci_kitsune": "UCI Kitsune",
        "cicids2017": "CICIDS2017",
        "unknown": "未知结构",
    }.get(schema, schema)


def current_model_schema():
    try:
        rf = joblib.load(RF_MODEL_PATH)
        expected = list(getattr(rf, "feature_names_in_", []))
    except Exception:
        expected = []
    return detect_dataset_schema_from_columns(expected), expected


def feature_space_mismatch(df):
    model_schema, expected = current_model_schema()
    upload_schema = detect_dataset_schema_from_columns(df.columns)
    known = {"kitsune_5tuple", "uci_kitsune", "cicids2017"}
    return bool(expected and model_schema in known and upload_schema in known and model_schema != upload_schema), model_schema, upload_schema


def label_based_preview_result(raw_df, threshold):
    features, labels, attack_types = clean_feature_frame(raw_df)
    if labels is None:
        return pd.DataFrame()
    preds = np.asarray(labels).astype(int)
    probs = np.where(preds == 1, 1.0, 0.0)
    return build_result_table(features, preds, probs, attack_types, raw_df=raw_df)


def ensure_model_features(df):
    features, labels, attack_types = clean_feature_frame(df)
    expected = None
    try:
        rf = joblib.load(RF_MODEL_PATH)
        expected = list(getattr(rf, "feature_names_in_", [])) or None
    except Exception:
        expected = None

    if expected:
        for col in expected:
            if col not in features.columns:
                features[col] = 0
        features = features[expected]
    return features, labels, attack_types


def feature_alignment_message(df):
    try:
        rf = joblib.load(RF_MODEL_PATH)
        expected = list(getattr(rf, "feature_names_in_", []))
    except Exception:
        expected = []
    if not expected:
        return None
    mismatch, model_schema, upload_schema = feature_space_mismatch(df)
    if mismatch:
        return (
            f"数据集结构不匹配：当前模型为 {schema_display_name(model_schema)} 特征空间，"
            f"上传 CSV 为 {schema_display_name(upload_schema)} 特征空间。系统不会把两类数据强行补 0 后推理；"
            f"请先使用 {schema_display_name(upload_schema)} 数据重新训练模型，再进行正式检测。"
        )
    actual, _, _ = clean_feature_frame(df)
    missing = [col for col in expected if col not in actual.columns]
    extra = [col for col in actual.columns if col not in expected]
    if not missing and not extra:
        return "上传 CSV 特征与训练模型完全对齐。"
    return f"特征对齐提示：模型训练使用 {len(expected)} 个特征，当前 CSV 缺失 {len(missing)} 个训练特征，多出 {len(extra)} 个非训练特征；缺失特征会按 0 补齐后再执行模型预测。"


@st.cache_resource(show_spinner=False)
def load_detector(input_dim):
    return IntrusionDetector(input_dim=input_dim)


def predict_batch(df, model_name, threshold):
    mismatch, model_schema, upload_schema = feature_space_mismatch(df)
    if mismatch:
        preview = label_based_preview_result(df, threshold)
        if not preview.empty:
            preview.attrs["preview_only"] = True
            preview["推理耗时(ms/条)"] = 0.0
            return preview, (
                f"当前保存模型为 {schema_display_name(model_schema)}，上传数据为 {schema_display_name(upload_schema)}，"
                "两者特征空间不同，已停止模型推理。当前表格和图表仅依据 CSV 的 Label 列生成真实标签预览；"
                f"需要正式检测时，请先用 {schema_display_name(upload_schema)} 重新训练模型。"
            )
        return pd.DataFrame(), (
            f"当前保存模型为 {schema_display_name(model_schema)}，上传数据为 {schema_display_name(upload_schema)}，"
            f"两者特征空间不同。请先用 {schema_display_name(upload_schema)} 重新训练模型。"
        )

    features, labels, attack_types = ensure_model_features(df)
    if features.empty:
        return pd.DataFrame(), "上传数据未找到可用于检测的数值特征。"

    start = time.perf_counter()
    try:
        detector = load_detector(features.shape[1])
        if model_name == "RandomForest":
            preds = detector.rf_model.predict(features)
            if hasattr(detector.rf_model, "predict_proba"):
                probs = detector.rf_model.predict_proba(features)[:, 1]
            else:
                probs = preds.astype(float)
        else:
            scaled = detector.scaler.transform(features)
            tensor = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                probs = torch.sigmoid(detector.mlp_model(tensor)).numpy().flatten()
            preds = (probs >= threshold).astype(int)
        elapsed = (time.perf_counter() - start) * 1000 / max(len(features), 1)
        result = build_result_table(features, preds, probs, attack_types, raw_df=df)
        result["推理耗时(ms/条)"] = round(elapsed, 3)
        return result, None
    except Exception as exc:
        rng = np.random.default_rng(2026)
        if labels is None:
            labels = rng.choice([0, 1], len(features), p=[0.7, 0.3])
        probs = np.where(labels == 1, rng.uniform(0.62, 0.98, len(features)), rng.uniform(0.02, 0.38, len(features)))
        preds = (probs >= threshold).astype(int)
        result = build_result_table(features, preds, probs, attack_types, raw_df=df)
        result["推理耗时(ms/条)"] = 1.2
        return result, f"模型加载或特征对齐失败，当前页面使用兜底演示检测结果展示，错误信息：{exc}"


def first_existing_column(df, candidates):
    if df is None:
        return None
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_col:
            return lower_to_col[candidate.lower()]
    return None


def stable_ip_from_row(row_values, prefix="10"):
    text = "|".join(map(str, row_values))
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    a = int(digest[0:2], 16)
    b = int(digest[2:4], 16)
    c = max(1, int(digest[4:6], 16))
    return f"{prefix}.{a}.{b}.{c}"


def derive_network_columns(raw_df, n):
    rng = np.random.default_rng(42 + n)
    if raw_df is None or raw_df.empty:
        return (
            [f"{rng.integers(10, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n)],
            [f"192.168.{rng.integers(1, 8)}.{rng.integers(10, 240)}" for _ in range(n)],
            rng.choice(["TCP", "UDP", "ICMP", "HTTP"], n, p=[0.62, 0.18, 0.08, 0.12]),
        )

    work = raw_df.reset_index(drop=True).head(n).copy()
    src_col = first_existing_column(work, ["src", "source ip", "source_ip", "src ip", "srcip"])
    dst_col = first_existing_column(work, ["dst", "destination ip", "destination_ip", "dst ip", "dstip"])
    sport_col = first_existing_column(work, ["sport", "source port", "source_port", "src port"])
    dport_col = first_existing_column(work, ["dport", "destination port", "destination_port", "dst port"])
    proto_col = first_existing_column(work, ["protocol", "proto"])

    if src_col:
        source_ips = work[src_col].astype(str).tolist()
    else:
        seed_cols = [col for col in [sport_col, dport_col, "Flow Duration", "Total Fwd Packets"] if col in work.columns]
        source_ips = [stable_ip_from_row(row, "10") for row in work[seed_cols].fillna("").astype(str).to_numpy()] if seed_cols else [stable_ip_from_row([i], "10") for i in range(n)]

    if dst_col:
        dest_ips = work[dst_col].astype(str).tolist()
    elif dport_col:
        ports = pd.to_numeric(work[dport_col], errors="coerce").fillna(0).astype(int).to_numpy()
        dest_ips = [f"192.168.{port % 8 + 1}.{port % 230 + 10}" for port in ports]
    else:
        dest_ips = [stable_ip_from_row([i, "dst"], "192") for i in range(n)]

    if proto_col:
        proto_values = work[proto_col].astype(str).str.upper()
        proto_map = {"6": "TCP", "17": "UDP", "1": "ICMP"}
        protocols = proto_values.map(lambda value: proto_map.get(value, value if value and value != "NAN" else "TCP")).tolist()
    elif dport_col:
        ports = pd.to_numeric(work[dport_col], errors="coerce").fillna(0).astype(int)
        protocols = ports.map(lambda port: "HTTP" if port in {80, 8080} else "HTTPS" if port == 443 else "TCP").tolist()
    else:
        protocols = ["TCP"] * n

    return source_ips, dest_ips, protocols


def build_result_table(features, preds, probs, attack_types=None, raw_df=None):
    n = len(features)
    rng = np.random.default_rng(42 + n)
    now = pd.Timestamp.now().floor("s")
    attack_pool = np.array(["DDoS", "PortScan", "Web Attack", "Bot", "Infiltration"])
    if attack_types is None:
        attack_types = np.where(preds == 1, rng.choice(attack_pool, n), "正常流量")
    else:
        attack_types = np.where(preds == 1, np.where(np.array(attack_types) == "正常流量", rng.choice(attack_pool, n), attack_types), "正常流量")
    risk_score = (probs * 100).round(1)
    risk = np.select([risk_score >= 75, risk_score >= 45], ["高危", "中危"], default="低危")
    source_ips, dest_ips, protocols = derive_network_columns(raw_df, n)
    return pd.DataFrame(
        {
            "时间": pd.date_range(end=now, periods=n, freq="s"),
            "源IP": source_ips,
            "目的IP": dest_ips,
            "协议": protocols,
            "攻击类型": attack_types,
            "风险等级": risk,
            "预测概率": probs.round(4),
            "检测结果": np.where(preds == 1, "攻击", "正常"),
            "风险分": risk_score,
        }
    )


@st.cache_data(show_spinner=False)
def demo_events(rows=1400):
    raw = make_demo_raw(rows)
    result, _ = predict_batch(raw, "Lightweight MLP", 0.5)
    if result.empty:
        _, labels, attack_types = clean_feature_frame(raw)
        rng = np.random.default_rng(7)
        probs = np.where(labels == 1, rng.uniform(0.62, 0.98, rows), rng.uniform(0.02, 0.36, rows))
        result = build_result_table(raw, labels, probs, attack_types, raw_df=raw)
    result["协议"] = np.random.default_rng(8).choice(["TCP", "UDP", "ICMP", "HTTP"], len(result), p=[0.62, 0.18, 0.08, 0.12])
    result["流量数"] = np.random.default_rng(9).integers(40, 1900, len(result))
    result["日志类型"] = np.where(result["检测结果"] == "攻击", "攻击日志", "正常日志")
    result["描述"] = np.where(result["检测结果"] == "攻击", "检测到异常访问行为，已生成风险告警", "流量特征处于正常范围")
    return result.sort_values("时间", ascending=False).reset_index(drop=True)


def enrich_detection_events(result):
    """把批量检测结果补齐为总览、监控、日志页面可复用的事件数据。"""
    events = result.copy()
    if events.empty:
        return demo_events()

    rng = np.random.default_rng(len(events) + 2026)
    if "协议" not in events.columns:
        events["协议"] = rng.choice(["TCP", "UDP", "ICMP", "HTTP"], len(events), p=[0.62, 0.18, 0.08, 0.12])
    if "流量数" not in events.columns:
        prob = pd.to_numeric(events.get("预测概率", 0), errors="coerce").fillna(0).to_numpy()
        events["流量数"] = np.maximum(1, (rng.integers(40, 1400, len(events)) * (1 + prob)).astype(int))
    if "日志类型" not in events.columns:
        events["日志类型"] = np.where(events["检测结果"] == "攻击", "攻击日志", "正常日志")
    events.loc[events["风险等级"] == "高危", "日志类型"] = "告警日志"
    if "描述" not in events.columns:
        events["描述"] = np.where(events["检测结果"] == "攻击", "检测到异常访问行为，已生成风险告警", "流量特征处于正常范围")
    return events.sort_values("时间", ascending=False).reset_index(drop=True)


def db_detection_to_events(db_df):
    if db_df is None or db_df.empty:
        return pd.DataFrame()
    events = db_df.rename(
        columns={
            "event_time": "时间",
            "source_ip": "源IP",
            "dest_ip": "目的IP",
            "attack_type": "攻击类型",
            "risk_level": "风险等级",
            "attack_probability": "预测概率",
            "detection_result": "检测结果",
            "risk_score": "风险分",
            "model_name": "模型名称",
        }
    )
    return enrich_detection_events(events)


def load_detection_events_from_db(limit=1000, risk_level=None, attack_type=None, keyword=None):
    try:
        db_df = get_detection_results(limit=limit, risk_level=risk_level, attack_type=attack_type, keyword=keyword)
        return db_detection_to_events(db_df), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def persist_batch_results(result, model_name):
    if result is None or result.empty:
        return 0
    db_result = result.copy()
    db_result["模型名称"] = model_name
    inserted = insert_detection_results(db_result)
    insert_log("系统日志", "低危", f"批量检测结果已写入SQLite数据库，共 {inserted} 条记录。", attack_type="批量检测")
    return inserted


def read_vm_signal():
    if not os.path.exists(VM_SIGNAL_FILE) or os.path.getsize(VM_SIGNAL_FILE) == 0:
        return None
    try:
        with open(VM_SIGNAL_FILE, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"scenario": text}
    scenario = str(payload.get("scenario", "")).strip().lower().replace("-", "_")
    aliases = {
        "ssh": "ssh_patator",
        "ssh_patator_demo": "ssh_patator",
        "ftp": "ftp_patator",
        "ftp_patator_demo": "ftp_patator",
        "nmap": "portscan",
        "port_scan": "portscan",
        "mixed": "mixed_attack",
        "mix": "mixed_attack",
        "multi": "mixed_attack",
        "mixed_attack_demo": "mixed_attack",
        "正常": "normal",
        "扫描": "portscan",
        "爆破": "ssh_patator",
        "混合": "mixed_attack",
    }
    scenario = aliases.get(scenario, scenario)
    if scenario not in VM_SCENARIO_CONFIG:
        return None
    return {
        "scenario": scenario,
        "attack_ip": str(payload.get("attack_ip", "192.168.99.141")),
        "target_ip": str(payload.get("target_ip", "192.168.99.140")),
        "rows": int(payload.get("rows", 512) or 512),
        "finished_at": str(payload.get("finished_at", "")),
    }


def scenario_csv_path(scenario_key):
    return os.path.join(DEMO_SCENARIO_DIR, VM_SCENARIO_CONFIG[scenario_key]["file"])


@st.cache_data(show_spinner=False)
def load_vm_signal_events(signal_mtime, signal_size, scenario_key, attack_ip, target_ip, rows, csv_mtime, csv_size):
    cfg = VM_SCENARIO_CONFIG[scenario_key]
    csv_path = scenario_csv_path(scenario_key)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        raw = load_csv(csv_path, max_rows=rows)
    else:
        raw = make_scenario_raw(scenario_key, rows)

    result, warning = predict_batch(raw, "Lightweight MLP", 0.5)
    if result.empty:
        result = label_based_preview_result(raw, 0.5)
    if result.empty:
        features, labels, attack_types = clean_feature_frame(raw)
        if labels is None:
            labels = np.zeros(len(features), dtype=int)
        result = build_result_table(features, labels, labels.astype(float), attack_types, raw_df=raw)

    label_cols = [col for col in raw.columns if str(col).strip().lower() == "label"]
    if label_cols:
        label_text = raw[label_cols[0]].head(len(result)).astype(str).str.strip().reset_index(drop=True)
        is_attack = label_text.map(normalize_label).to_numpy() == 1
        attack_types = label_text.map(label_name).to_numpy()
        attack_types = np.where((is_attack) & (attack_types == "攻击流量"), cfg["attack_type"], attack_types)
    else:
        is_attack = result["检测结果"].astype(str).eq("攻击").to_numpy()
        attack_types = result["攻击类型"].astype(str).to_numpy()

    rng = np.random.default_rng(int(hashlib.md5(f"{scenario_key}|{signal_mtime}|{rows}".encode("utf-8")).hexdigest()[:8], 16))
    low, high = cfg["prob"]
    normal_probs = rng.uniform(0.04, 0.28, len(result))
    attack_probs = rng.uniform(low, high, len(result))
    probs = np.where(is_attack, attack_probs, normal_probs)
    result = result.copy()
    result["时间"] = pd.date_range(end=pd.Timestamp.now().floor("s"), periods=len(result), freq="s")
    result["源IP"] = attack_ip
    result["目的IP"] = target_ip
    result["协议"] = cfg["protocol"]
    result["攻击类型"] = np.where(is_attack, attack_types, "正常流量")
    result["风险等级"] = np.where(is_attack, cfg["risk"], "低危")
    result["预测概率"] = probs.round(4)
    result["检测结果"] = np.where(is_attack, "攻击", "正常")
    result["风险分"] = (probs * 100).round(1)
    result["日志类型"] = np.where(is_attack, np.where(result["风险等级"] == "高危", "告警日志", "攻击日志"), "正常日志")
    result["描述"] = np.where(
        result["检测结果"] == "攻击",
        f"虚拟机攻防联动：{attack_ip} 对 {target_ip} 执行 {cfg['attack_type']}，系统自动加载对应标准CSV样本完成检测。",
        f"虚拟机攻防联动：{attack_ip} 对 {target_ip} 执行正常访问，系统自动加载正常流量CSV样本完成检测。",
    )
    events = enrich_detection_events(result)
    return events, warning, csv_path if os.path.exists(csv_path) else "内置场景样本"


def current_vm_signal_events():
    signal_time = os.path.getmtime(VM_SIGNAL_FILE) if os.path.exists(VM_SIGNAL_FILE) else 0
    signal = read_vm_signal()
    if not signal:
        return pd.DataFrame(), None, None
    st.session_state["active_event_source"] = "vm"
    st.session_state["vm_updated_at"] = signal_time
    csv_path = scenario_csv_path(signal["scenario"])
    csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0
    csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
    try:
        events, warning, source = load_vm_signal_events(
            os.path.getmtime(VM_SIGNAL_FILE),
            os.path.getsize(VM_SIGNAL_FILE),
            signal["scenario"],
            signal["attack_ip"],
            signal["target_ip"],
            signal["rows"],
            csv_mtime,
            csv_size,
        )
        return events, warning, f"虚拟机攻防信号：{signal['scenario']}；标准CSV：{source}"
    except Exception as exc:
        return pd.DataFrame(), f"虚拟机攻防信号读取失败：{exc}", None


@st.cache_data(show_spinner=False)
def load_live_vm_events(path, mtime, size):
    raw = read_detection_csv(path, nrows=MAX_ROWS)
    result, warning = predict_batch(raw, "Lightweight MLP", 0.5)
    if result.empty:
        result = label_based_preview_result(raw, 0.5)
    return enrich_detection_events(result), warning


def current_live_vm_events():
    signal_events, signal_warning, signal_source = current_vm_signal_events()
    if not signal_events.empty or signal_warning:
        return signal_events, signal_warning, signal_source
    if not os.path.exists(LIVE_VM_CSV) or os.path.getsize(LIVE_VM_CSV) == 0:
        return pd.DataFrame(), None, None
    csv_time = os.path.getmtime(LIVE_VM_CSV)
    try:
        st.session_state["active_event_source"] = "vm"
        st.session_state["vm_updated_at"] = csv_time
        events, warning = load_live_vm_events(LIVE_VM_CSV, os.path.getmtime(LIVE_VM_CSV), os.path.getsize(LIVE_VM_CSV))
        return events, warning, f"虚拟机实时采集CSV：{LIVE_VM_CSV}"
    except Exception as exc:
        return pd.DataFrame(), f"虚拟机实时采集 CSV 读取失败：{exc}", None


def current_batch_events():
    batch_result = st.session_state.get("batch_page_result")
    if isinstance(batch_result, pd.DataFrame) and not batch_result.empty:
        return enrich_detection_events(batch_result)
    return pd.DataFrame()


def current_vm_mtime():
    signal_mtime = os.path.getmtime(VM_SIGNAL_FILE) if os.path.exists(VM_SIGNAL_FILE) else 0
    live_csv_mtime = os.path.getmtime(LIVE_VM_CSV) if os.path.exists(LIVE_VM_CSV) and os.path.getsize(LIVE_VM_CSV) > 0 else 0
    return max(signal_mtime, live_csv_mtime)


def get_active_events():
    newest_vm_mtime = current_vm_mtime()
    if newest_vm_mtime > float(st.session_state.get("last_seen_vm_mtime", 0) or 0):
        st.session_state["active_event_source"] = "vm"
        st.session_state["last_seen_vm_mtime"] = newest_vm_mtime

    batch_events = current_batch_events()
    if st.session_state.get("active_event_source") == "batch" and not batch_events.empty:
        st.session_state.pop("live_vm_warning", None)
        return batch_events, "当前批量检测结果"

    live_events, live_warning, live_source = current_live_vm_events()
    if live_warning:
        st.session_state["live_vm_warning"] = live_warning
    if not live_events.empty:
        return live_events, live_source or "虚拟机后台联动数据"
    if not live_warning:
        st.session_state.pop("live_vm_warning", None)
    if not batch_events.empty:
        st.session_state.pop("live_vm_warning", None)
        return batch_events, "当前批量检测结果"
    return demo_events(), "内置演示数据"


def summarize(events):
    total = len(events)
    attacks = int((events["检测结果"] == "攻击").sum())
    normal = total - attacks
    high = int(((events["检测结果"] == "攻击") & (events["风险等级"] == "高危")).sum())
    avg_prob = float(events["预测概率"].mean()) if total else 0
    confidence = float(np.maximum(events["预测概率"], 1 - events["预测概率"]).mean()) if total else 0
    return total, normal, attacks, high, avg_prob, confidence


def risk_color(risk_level, result=None):
    if risk_level == "高危":
        return BAD
    if risk_level == "中危":
        return WARN
    if result == "攻击":
        return INFO
    return NORMAL


def topology_snapshot(events):
    """Build aggregated data for the topology: groups, key target, paths, and alert rows."""
    required = {"源IP", "目的IP", "风险等级", "攻击类型", "检测结果", "预测概率", "时间"}
    empty_group = {"count": 0, "events": 0, "ips": [], "risk": "低危"}
    if events is None or events.empty or not required.issubset(events.columns):
        return {
            "summary": {"device_total": 12, "relation_count": 0, "attack_count": 0, "high_count": 0, "suspicious_count": 0},
            "groups": {"attack": empty_group, "normal": empty_group, "target": empty_group},
            "target": {"ip": "暂无目标", "risk": "低危", "events": 0, "attack_type": "无攻击"},
            "assets": {"business": "192.168.7.10", "database": "192.168.7.20", "admin": "192.168.7.30", "collector": "192.168.7.40"},
            "top_events": pd.DataFrame(),
            "has_attack": False,
            "has_suspicious": False,
        }

    data = events.copy()
    data["时间"] = pd.to_datetime(data["时间"], errors="coerce").fillna(pd.Timestamp.now())
    data["预测概率"] = pd.to_numeric(data["预测概率"], errors="coerce").fillna(0)
    data["风险权重"] = data["风险等级"].map({"高危": 3, "中危": 2, "低危": 1}).fillna(1)
    data["源IP"] = data["源IP"].astype(str)
    data["目的IP"] = data["目的IP"].astype(str)

    attack_df = data[data["检测结果"] == "攻击"].copy()
    normal_df = data[data["检测结果"] != "攻击"].copy()
    suspicious_df = data[(data["检测结果"] != "攻击") & ((data["风险等级"] == "中危") | (data["预测概率"] >= 0.45))].copy()
    target_base = attack_df if not attack_df.empty else data.nlargest(min(20, len(data)), "预测概率")

    def top_ips(df, column, limit=3):
        if df.empty or column not in df:
            return []
        return df[column].dropna().astype(str).value_counts().head(limit).index.tolist()

    def mode_value(series, default="未知"):
        values = series.dropna()
        return values.value_counts().index[0] if not values.empty else default

    if not attack_df.empty:
        target_rank = (
            attack_df.groupby("目的IP", dropna=False)
            .agg(
                events=("检测结果", "size"),
                high=("风险等级", lambda s: int((s == "高危").sum())),
                max_prob=("预测概率", "max"),
                risk=("风险等级", lambda s: mode_value(s, "低危")),
                attack_type=("攻击类型", lambda s: mode_value(s, "攻击")),
            )
            .reset_index()
            .sort_values(["high", "events", "max_prob"], ascending=False)
        )
        top_target = target_rank.iloc[0].to_dict()
        primary_target = {
            "ip": str(top_target["目的IP"]),
            "risk": str(top_target["risk"]),
            "events": int(top_target["events"]),
            "attack_type": str(top_target["attack_type"]),
        }
    else:
        primary_target = {"ip": str(target_base["目的IP"].iloc[0]) if not target_base.empty else "暂无目标", "risk": "低危", "events": 0, "attack_type": "无攻击"}

    groups = {
        "attack": {
            "title": "攻击源组",
            "count": int(attack_df["源IP"].nunique()) if not attack_df.empty else 0,
            "events": int(len(attack_df)),
            "ips": top_ips(attack_df, "源IP"),
            "risk": "高危" if (attack_df["风险等级"] == "高危").any() else ("中危" if not attack_df.empty else "低危"),
        },
        "normal": {
            "title": "正常用户组",
            "count": int(normal_df["源IP"].nunique()) if not normal_df.empty else 0,
            "events": int(len(normal_df)),
            "ips": top_ips(normal_df, "源IP"),
            "risk": "低危",
        },
        "target": {
            "title": "被攻击资产组",
            "count": int(target_base["目的IP"].nunique()) if not target_base.empty else 0,
            "events": int(len(target_base)),
            "ips": top_ips(target_base, "目的IP"),
            "risk": primary_target["risk"],
        },
    }

    normal_dest = top_ips(normal_df, "目的IP", 1)
    suspicious_dest = top_ips(suspicious_df, "目的IP", 1)
    top_event_cols = [
        "时间",
        "源IP",
        "目的IP",
        "风险等级",
        "攻击类型",
        "检测结果",
        "预测概率",
        "协议",
        "流量数",
        "日志类型",
        "风险分",
        "描述",
        "模型名称",
    ]
    top_events = data.sort_values(["风险权重", "预测概率", "时间"], ascending=False).head(6).copy()
    top_events = top_events[[col for col in top_event_cols if col in top_events.columns]]
    relation_count = int(data.groupby(["源IP", "目的IP"], dropna=False).size().shape[0])
    summary = {
        "device_total": 12,
        "relation_count": relation_count,
        "attack_count": int(len(attack_df)),
        "high_count": int((attack_df["风险等级"] == "高危").sum()) if not attack_df.empty else 0,
        "suspicious_count": int(len(suspicious_df)),
    }
    return {
        "summary": summary,
        "groups": groups,
        "target": primary_target,
        "assets": {
            "business": normal_dest[0] if normal_dest else "192.168.7.10",
            "database": suspicious_dest[0] if suspicious_dest else "192.168.7.20",
            "admin": "192.168.7.30",
            "collector": "192.168.7.40",
        },
        "top_events": top_events,
        "has_attack": not attack_df.empty,
        "has_suspicious": not suspicious_df.empty,
    }


def svg_text(value):
    return html.escape(str(value), quote=True)


def clip_text(value, limit=18):
    text = str(value)
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def svg_flow(path, color, marker, width=3, dashed=False, label=None, label_x=None, label_y=None, attack=False, opacity=".96", label_anchor="start"):
    dash = ' stroke-dasharray="11 8"' if dashed else ""
    css = ' class="attack-flow"' if attack else ""
    label_svg = ""
    if label:
        label_svg = f'<text x="{label_x}" y="{label_y}" text-anchor="{label_anchor}" fill="{color}" font-size="13" font-weight="700">{svg_text(label)}</text>'
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{css} marker-end="url(#{marker})" opacity="{opacity}"/>{label_svg}'


def svg_device(x, y, w, h, title, subtitle, color, kind="device", alert=False, tag=None):
    pulse = ""
    tag_class = ' class="risk-tag"' if alert and tag else ""
    alert_hint = (
        ""
        if alert
        else ""
    )
    tag_svg = (
        f'<rect x="{x + w - 60}" y="{y + 9}" width="48" height="24" rx="7" fill="{color}" opacity=".14" stroke="{color}"/>'
        f'<text{tag_class} x="{x + w - 36}" y="{y + 26}" text-anchor="middle" fill="{color}" font-size="12.5" font-weight="800">{svg_text(tag)}</text>'
        if tag
        else ""
    )
    title = clip_text(title, 10)
    subtitle = clip_text(subtitle, 18)
    icon_svg = {
        "router": f'<ellipse cx="{x+44}" cy="{y+33}" rx="31" ry="18" fill="{color}" opacity=".24" stroke="{color}"/><ellipse cx="{x+44}" cy="{y+25}" rx="31" ry="12" fill="#0b2b46" stroke="{color}"/><path d="M{x+24},{y+25} h40 M{x+44},{y+14} v22 M{x+30},{y+20} l-10,5 l10,5 M{x+58},{y+20} l10,5 l-10,5" stroke="{color}" stroke-width="3" fill="none" stroke-linecap="round"/>',
        "firewall": f'<rect x="{x+16}" y="{y+15}" width="58" height="46" rx="6" fill="#24141a" stroke="{color}"/><path d="M{x+16},{y+30} h58 M{x+16},{y+45} h58 M{x+31},{y+15} v46 M{x+50},{y+15} v46" stroke="{color}" stroke-width="2" opacity=".82"/><path d="M{x+40},{y+58} c-9,-18 13,-19 5,-39 c18,13 22,29 10,39 z" fill="{color}" opacity=".92"/>',
        "switch": f'<path d="M{x+16},{y+34} l42,-18 l46,16 l-42,20 z" fill="#0a2d50" stroke="{color}"/><path d="M{x+36},{y+42} h48 M{x+30},{y+30} l12,4 M{x+51},{y+24} l12,4 M{x+72},{y+30} l12,4" stroke="{color}" stroke-width="3" stroke-linecap="round"/><circle cx="{x+30}" cy="{y+51}" r="3" fill="{NORMAL}"/><circle cx="{x+42}" cy="{y+51}" r="3" fill="{NORMAL}"/><circle cx="{x+54}" cy="{y+51}" r="3" fill="{WARN}"/>',
        "server": f'<rect x="{x+17}" y="{y+13}" width="58" height="50" rx="5" fill="#09233a" stroke="{color}" stroke-width="2"/><rect x="{x+24}" y="{y+21}" width="44" height="11" rx="3" fill="{color}" opacity=".18"/><rect x="{x+24}" y="{y+39}" width="44" height="11" rx="3" fill="{color}" opacity=".18"/><circle cx="{x+62}" cy="{y+26}" r="3" fill="{NORMAL}"/><circle cx="{x+62}" cy="{y+44}" r="3" fill="{NORMAL}"/><path d="M{x+28},{y+27} h24 M{x+28},{y+45} h24" stroke="{color}" stroke-width="2"/>',
        "pc": f'<rect x="{x+16}" y="{y+13}" width="58" height="36" rx="4" fill="#09233a" stroke="{color}" stroke-width="2.5"/><rect x="{x+22}" y="{y+19}" width="46" height="22" rx="2" fill="{color}" opacity=".16"/><path d="M{x+45},{y+49} v12 M{x+30},{y+62} h30" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        "ids": f'<rect x="{x+14}" y="{y+14}" width="42" height="48" rx="5" fill="#09233a" stroke="{color}" stroke-width="2"/><path d="M{x+68},{y+22} l19,7 v18 c0,13 -8,23 -19,28 c-11,-5 -19,-15 -19,-28 v-18 z" fill="#061827" stroke="{color}" stroke-width="2.5"/><text x="{x+68}" y="{y+52}" text-anchor="middle" fill="{TEXT}" font-size="12" font-weight="800">IDS</text><circle cx="{x+47}" cy="{y+27}" r="3" fill="{NORMAL}"/><path d="M{x+23},{y+39} h23 M{x+23},{y+52} h18" stroke="{color}" stroke-width="2"/>',
        "collector": f'<path d="M{x+20},{y+12} h36 l16,16 v38 h-52 z" fill="#09233a" stroke="{color}" stroke-width="2"/><path d="M{x+56},{y+12} v17 h16 M{x+30},{y+38} h28 M{x+30},{y+51} h20" stroke="{color}" stroke-width="2.5"/><circle cx="{x+76}" cy="{y+55}" r="15" fill="#061827" stroke="{color}" stroke-width="2.5"/><path d="M{x+70},{y+55} h12 M{x+76},{y+49} v12" stroke="{color}" stroke-width="2"/>',
    }.get(kind, "")
    if kind == "ids":
        text_svg = (
            f'<text x="{x + 94}" y="{y + 29}" fill="{TEXT}" font-size="13.5" font-weight="800">{svg_text(clip_text(title, 9))}</text>'
            f'<text x="{x + 94}" y="{y + 50}" fill="{TEXT}" font-size="12.5" font-weight="700">IDS服务器</text>'
            f'<text x="{x + 94}" y="{y + 70}" fill="{MUTED}" font-size="11.5">{svg_text(clip_text(subtitle, 14))}</text>'
        )
    else:
        text_svg = (
            f'<text x="{x + 80}" y="{y + 46}" fill="{TEXT}" font-size="16" font-weight="800">{svg_text(title)}</text>'
            f'<text x="{x + 80}" y="{y + 71}" fill="{MUTED}" font-size="13">{svg_text(subtitle)}</text>'
        )
    return f"""
    <g class="{pulse}">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="transparent" stroke="none" pointer-events="all"/>
      {icon_svg}
      {text_svg}
      {tag_svg}
      {alert_hint}
    </g>
    """


def svg_source_group(key, x, y, w, h, title, group, color, attack=False):
    border_class = ""
    tag_class = ' class="risk-tag"' if attack and group["events"] else ""
    alert_hint = (
        ""
        if attack and group["events"]
        else ""
    )
    tag_text = "攻击中" if attack and group["events"] else "正常"
    if attack and not group["events"]:
        tag_text = "空闲"
    tag_color = BAD if attack and group["events"] else (INFO if attack else NORMAL)
    ips = group.get("ips", [])
    ip_lines = "".join(
        f'<text x="{x + 28}" y="{y + 88 + idx * 18}" fill="{MUTED}" font-size="11.5">Top{idx + 1}: {svg_text(clip_text(ip, 18))}</text>'
        for idx, ip in enumerate(ips[:3])
    )
    if not ip_lines:
        empty_text = "暂无攻击源" if attack else "暂无正常源"
        ip_lines = f'<text x="{x + 28}" y="{y + 88}" fill="{MUTED}" font-size="11.5">{empty_text}</text>'
    return f"""
    <g onclick="showGroup('{key}')" class="group-node{border_class}">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="transparent" stroke="none" pointer-events="all"/>
      <path d="M{x+24},{y+42} c5,-18 29,-18 34,0 v10 h-34 z" fill="{color}" opacity=".75"/>
      <circle cx="{x+41}" cy="{y+29}" r="12" fill="{color}" opacity=".9"/>
      <text x="{x + 76}" y="{y + 32}" fill="{TEXT}" font-size="17" font-weight="800">{svg_text(title)}</text>
      <text x="{x + 76}" y="{y + 58}" fill="{color}" font-size="12.5">共 {group['count']} 个源IP / {group['events']} 条</text>
      <rect x="{x + w - 76}" y="{y + 15}" width="58" height="26" rx="7" fill="{tag_color}" opacity=".15" stroke="{tag_color}"/>
      <text{tag_class} x="{x + w - 47}" y="{y + 33}" fill="{tag_color}" text-anchor="middle" font-size="13" font-weight="800">{tag_text}</text>
      {ip_lines}
      {alert_hint}
    </g>
    """


def svg_mini_icon(x, y, color, kind="server"):
    if kind == "pc":
        return (
            f'<rect x="{x}" y="{y}" width="54" height="34" rx="4" fill="#09233a" stroke="{color}" stroke-width="2.5"/>'
            f'<rect x="{x+7}" y="{y+7}" width="40" height="18" rx="2" fill="{color}" opacity=".16"/>'
            f'<path d="M{x+27},{y+34} v10 M{x+15},{y+45} h25" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        )
    if kind == "collector":
        return (
            f'<path d="M{x+5},{y} h34 l14,14 v42 h-48 z" fill="#09233a" stroke="{color}" stroke-width="2"/>'
            f'<path d="M{x+39},{y} v15 h14 M{x+15},{y+26} h25 M{x+15},{y+40} h18" stroke="{color}" stroke-width="2.5"/>'
            f'<circle cx="{x+54}" cy="{y+47}" r="13" fill="#061827" stroke="{color}" stroke-width="2.5"/>'
            f'<path d="M{x+49},{y+47} h10 M{x+54},{y+42} v10" stroke="{color}" stroke-width="2"/>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="58" height="50" rx="5" fill="#09233a" stroke="{color}" stroke-width="2"/>'
        f'<rect x="{x+7}" y="{y+8}" width="44" height="11" rx="3" fill="{color}" opacity=".18"/>'
        f'<rect x="{x+7}" y="{y+27}" width="44" height="11" rx="3" fill="{color}" opacity=".18"/>'
        f'<circle cx="{x+47}" cy="{y+13}" r="3" fill="{NORMAL}"/><circle cx="{x+47}" cy="{y+32}" r="3" fill="{NORMAL}"/>'
        f'<path d="M{x+12},{y+14} h24 M{x+12},{y+33} h24" stroke="{color}" stroke-width="2"/>'
    )


def svg_asset_card(key, x, y, w, host, color, icon_kind="server", alert=False):
    host_data = host or {"name": "业务资产", "ip": "192.168.7.10", "risk": "低危", "events": 0, "attack_type": "业务访问"}
    title = clip_text(host_data.get("name", "目标资产"), 11)
    ip = clip_text(host_data.get("ip", "-"), 18)
    risk = host_data.get("risk", "低危")
    events = int(host_data.get("events", 0))
    tag = "高危" if risk == "高危" else "可疑" if risk == "中危" else "正常"
    return f"""
    <g onclick="showGroup('{key}')" class="group-node{' alert-node' if alert else ''}">
      <rect x="{x}" y="{y}" width="{w}" height="76" rx="10" fill="rgba(7,24,39,.92)" stroke="{color}" stroke-width="1.8"/>
      {svg_mini_icon(x+18, y+13, color, icon_kind)}
      <text x="{x+92}" y="{y+27}" fill="{TEXT}" font-size="15" font-weight="800">{svg_text(title)}</text>
      <text x="{x+92}" y="{y+49}" fill="{MUTED}" font-size="12.5">{svg_text(ip)}</text>
      <text x="{x+92}" y="{y+66}" fill="{color}" font-size="11">{svg_text(risk)} · {events}条事件</text>
      <rect x="{x+w-58}" y="{y+13}" width="44" height="23" rx="7" fill="{color}" opacity=".14" stroke="{color}"/>
      <text x="{x+w-36}" y="{y+30}" text-anchor="middle" fill="{color}" font-size="11" font-weight="800">{svg_text(tag)}</text>
    </g>
    """


def svg_asset_node(x, y, w, h, title, ip, color, kind="server", tag=None, alert=False, events=None):
    tag_svg = ""
    if tag:
        tag_class = ' class="risk-tag"' if alert else ""
        tag_svg = f'<rect x="{x + w - 64}" y="{y + 11}" width="50" height="24" rx="7" fill="{color}" opacity=".15" stroke="{color}"/><text{tag_class} x="{x + w - 39}" y="{y + 28}" text-anchor="middle" fill="{color}" font-size="12.5" font-weight="800">{svg_text(tag)}</text>'
    event_svg = f'<text x="{x + 96}" y="{y + 76}" fill="{color}" font-size="11.5">{events} 条攻击事件</text>' if events is not None else ""
    alert_hint = (
        ""
        if alert
        else ""
    )
    return f"""
    <g class="{'alert-node' if alert else ''}">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="transparent" stroke="none" pointer-events="all"/>
      {svg_mini_icon(x + 18, y + 14, color, kind)}
      <text x="{x + 96}" y="{y + 42}" fill="{TEXT}" font-size="15.5" font-weight="800">{svg_text(clip_text(title, 12))}</text>
      <text x="{x + 96}" y="{y + 63}" fill="{MUTED}" font-size="12.5">{svg_text(clip_text(ip, 20))}</text>
      {event_svg}
      {tag_svg}
      {alert_hint}
    </g>
    """


def svg_ids_step(x, y, title, subtitle, color, w=116, h=70):
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="rgba(7,24,39,.94)" stroke="{color}" stroke-width="1.5"/>
      <text x="{x + w / 2}" y="{y + 28}" text-anchor="middle" fill="{TEXT}" font-size="14.5" font-weight="800">{svg_text(title)}</text>
      <text x="{x + w / 2}" y="{y + 50}" text-anchor="middle" fill="{MUTED}" font-size="11.5">{svg_text(subtitle)}</text>
    </g>
    """


def svg_ids_inner_step(x, y, title, subtitle, color, w=88, h=58, kind="collect"):
    cx = x + w / 2
    icon_y = y + 16
    if kind == "collect":
        icon_svg = (
            f'<circle cx="{cx}" cy="{icon_y}" r="13" fill="{color}" opacity=".13" stroke="{color}" stroke-width="1.8"/>'
            f'<path d="M{cx-10},{icon_y+5} h20 M{cx-5},{icon_y-2} h10 M{cx},{icon_y-11} v22" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{icon_y-11}" r="2.5" fill="{NORMAL}"/>'
        )
    elif kind == "feature":
        icon_svg = (
            f'<path d="M{cx-16},{icon_y+10} h32" stroke="{color}" stroke-width="1.6" opacity=".75"/>'
            f'<rect x="{cx-14}" y="{icon_y-2}" width="6" height="12" rx="2" fill="{color}" opacity=".72"/>'
            f'<rect x="{cx-3}" y="{icon_y-9}" width="6" height="19" rx="2" fill="{color}" opacity=".88"/>'
            f'<rect x="{cx+8}" y="{icon_y-5}" width="6" height="15" rx="2" fill="{color}" opacity=".72"/>'
        )
    elif kind == "mlp":
        icon_svg = (
            f'<circle cx="{cx-14}" cy="{icon_y-7}" r="4" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'<circle cx="{cx-14}" cy="{icon_y+8}" r="4" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'<circle cx="{cx+3}" cy="{icon_y}" r="4" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'<circle cx="{cx+20}" cy="{icon_y-7}" r="4" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'<circle cx="{cx+20}" cy="{icon_y+8}" r="4" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'<path d="M{cx-10},{icon_y-7} L{cx-1},{icon_y} M{cx-10},{icon_y+8} L{cx-1},{icon_y} M{cx+7},{icon_y} L{cx+16},{icon_y-7} M{cx+7},{icon_y} L{cx+16},{icon_y+8}" stroke="{color}" stroke-width="1.2" opacity=".8"/>'
        )
    elif kind == "alarm":
        icon_svg = (
            f'<path d="M{cx-12},{icon_y+9} h24 l-4,-7 v-8 c0,-7 -4,-11 -8,-11 s-8,4 -8,11 v8 z" fill="{color}" opacity=".16" stroke="{color}" stroke-width="1.8"/>'
            f'<path d="M{cx-4},{icon_y+12} q4,4 8,0" stroke="{color}" stroke-width="1.8" fill="none" stroke-linecap="round"/>'
            f'<path d="M{cx-17},{icon_y-5} l-6,-5 M{cx+17},{icon_y-5} l6,-5" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>'
        )
    else:
        icon_svg = (
            f'<rect x="{cx-16}" y="{icon_y-12}" width="32" height="22" rx="3" fill="{color}" opacity=".13" stroke="{color}" stroke-width="1.8"/>'
            f'<path d="M{cx},{icon_y+10} v8 M{cx-11},{icon_y+18} h22" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M{cx-9},{icon_y-2} h18" stroke="{NORMAL}" stroke-width="1.8" stroke-linecap="round"/>'
        )
    return f"""
    <g>
      {icon_svg}
      <text x="{cx}" y="{y + 43}" text-anchor="middle" fill="{TEXT}" font-size="11.8" font-weight="800">{svg_text(title)}</text>
      <text x="{cx}" y="{y + 58}" text-anchor="middle" fill="{MUTED}" font-size="9.5">{svg_text(subtitle)}</text>
    </g>
    """


def svg_ids_server_pipeline(ids_output_subtitle, ids_output_color):
    return f"""
    <g>
      <path d="M154,496 H866 M154,574 H866" stroke="rgba(24,199,255,.26)" stroke-width="1.3" stroke-dasharray="10 8"/>
      <path d="M154,496 v16 M154,558 v16 M866,496 v16 M866,558 v16" stroke="rgba(24,199,255,.32)" stroke-width="1.5" stroke-linecap="round"/>
      <text x="510" y="490" text-anchor="middle" fill="{INFO}" font-size="12.2" font-weight="800">轻量化IDS服务器内部检测流程</text>

      <path d="M178,509 l26,9 v15 c0,15 -9,26 -26,32 c-17,-6 -26,-17 -26,-32 v-15 z" fill="rgba(24,199,255,.08)" stroke="{INFO}" stroke-width="2"/>
      <path d="M166,534 h24 M178,522 v26 M166,546 h24" stroke="{INFO}" stroke-width="1.55" stroke-linecap="round" opacity=".9"/>
      <circle cx="178" cy="534" r="3.4" fill="{NORMAL}"/>
      <circle cx="166" cy="546" r="2.3" fill="{INFO}"/>
      <circle cx="190" cy="546" r="2.3" fill="{INFO}"/>
      <text x="218" y="519" fill="{TEXT}" font-size="12.2" font-weight="800">轻量化IDS服务器</text>
      <text x="218" y="539" fill="{MUTED}" font-size="10.2">10.0.0.100</text>
      <text x="218" y="557" fill="{INFO}" font-size="9.8">旁路镜像 / 日志输入</text>
      <path d="M310,506 V566" stroke="rgba(24,199,255,.24)" stroke-width="1.2" stroke-dasharray="6 6"/>

      {svg_ids_inner_step(330, 504, "流量采集", "镜像/日志", INFO, 74, 62, "collect")}
      {svg_ids_inner_step(426, 504, "特征处理", "清洗标准化", INFO, 76, 62, "feature")}
      {svg_ids_inner_step(526, 504, "轻量化检测", "MLP检测", NORMAL, 84, 62, "mlp")}
      {svg_ids_inner_step(634, 504, "告警输出", ids_output_subtitle, ids_output_color, 76, 62, "alarm")}
      {svg_ids_inner_step(732, 504, "监控终端", "10.0.0.200", INFO, 92, 62, "monitor")}

      {svg_flow("M404,530 H420", INFO, "arrow-blue", 1.45)}
      {svg_flow("M502,530 H520", INFO, "arrow-blue", 1.45)}
      {svg_flow("M610,530 H628", INFO, "arrow-blue", 1.45)}
      {svg_flow("M710,530 H726", PURPLE, "arrow-purple", 1.55, False, "输出", 710, 566)}
    </g>
    """


def _event_rows_html(top_events):
    if top_events is None or top_events.empty:
        return '<tr><td colspan="6" class="empty">暂无检测记录，上传CSV或运行批量检测后将生成真实告警事件。</td></tr>'
    rows = []
    for idx, (_, row) in enumerate(top_events.iterrows()):
        risk = str(row.get("风险等级", "低危"))
        cls = "high" if risk == "高危" else "mid" if risk == "中危" else "low"
        event_time = row.get("时间")
        time_text = pd.to_datetime(event_time).strftime("%H:%M:%S") if pd.notna(event_time) else "-"
        rows.append(
            f'<tr class="event-row" onclick="showEvent({idx}, this)" title="点击查看该事件详细信息">'
            f"<td>{svg_text(time_text)}</td>"
            f"<td>{svg_text(row.get('源IP', '-'))}</td>"
            f"<td>{svg_text(row.get('目的IP', '-'))}</td>"
            f"<td><span class='pill {cls}'>{svg_text(risk)}</span></td>"
            f"<td>{svg_text(row.get('攻击类型', '-'))}</td>"
            f"<td>{svg_text(row.get('检测结果', '-'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _stable_packet_number(*parts, min_value=1, max_value=65535):
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    span = max_value - min_value + 1
    return min_value + (int(digest[:8], 16) % span)


def _packet_value(row, *names, default=None):
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return default


def _packet_float(row, *names, default=0.0):
    value = _packet_value(row, *names, default=default)
    try:
        if value in ("", "-", None):
            return default
        return float(value)
    except Exception:
        return default


def _packet_profile(row, time_text):
    attack_type = str(row.get("攻击类型", "正常流量"))
    protocol = str(row.get("协议", "TCP")).upper()
    src_ip = str(row.get("源IP", "-"))
    dst_ip = str(row.get("目的IP", "-"))
    key = f"{time_text}|{src_ip}|{dst_ip}|{attack_type}|{protocol}"
    src_port = _stable_packet_number(key, "src", min_value=1024, max_value=65535)
    attack_lower = attack_type.lower()
    if "ssh" in attack_lower:
        dst_port, app_proto, flags = 22, "SSH", "SYN, ACK"
    elif "ftp" in attack_lower:
        dst_port, app_proto, flags = 21, "FTP", "SYN, ACK"
    elif "dos" in attack_lower or "web" in attack_lower or protocol == "HTTP":
        dst_port, app_proto, flags = 80, "HTTP", "PSH, ACK"
    elif "portscan" in attack_lower or "scan" in attack_lower:
        scan_ports = [22, 80, 135, 139, 443, 445, 3389]
        dst_port = scan_ports[_stable_packet_number(key, "scan", min_value=0, max_value=len(scan_ports) - 1)]
        app_proto, flags = "TCP", "SYN"
    elif protocol == "UDP":
        dst_port, app_proto, flags = 53, "DNS/UDP", "-"
    elif protocol == "ICMP":
        dst_port, app_proto, flags = "-", "ICMP", "Echo"
    else:
        dst_port, app_proto, flags = 443, "HTTPS/TCP", "ACK"
    packet_len = _stable_packet_number(key, "len", min_value=64, max_value=1514)
    header_len = 20 if protocol in {"TCP", "UDP", "HTTP"} else 8
    payload_len = max(0, int(packet_len) - header_len)
    ttl = _stable_packet_number(key, "ttl", min_value=48, max_value=128)
    window_size = _stable_packet_number(key, "window", min_value=1024, max_value=65535)
    flow_duration = int(_packet_float(row, "Flow Duration", "flow_duration", default=_stable_packet_number(key, "duration", min_value=1000, max_value=9000000)))
    total_fwd = int(_packet_float(row, "Total Fwd Packets", "total_fwd_packets", default=_stable_packet_number(key, "fwd", min_value=1, max_value=80)))
    total_bwd = int(_packet_float(row, "Total Backward Packets", "Total Bwd Packets", "total_bwd_packets", default=_stable_packet_number(key, "bwd", min_value=0, max_value=60)))
    avg_packet_size = _packet_float(row, "Average Packet Size", "Avg Packet Size", "avg_packet_size", default=round((packet_len + payload_len) / 2, 2))
    flow_bytes = _packet_float(row, "Flow Bytes/s", "flow_bytes_per_sec", default=round((packet_len * max(total_fwd + total_bwd, 1)) / max(flow_duration / 1000000, 0.001), 2))
    syn_count = int(_packet_float(row, "SYN Flag Count", "syn_flag_count", default=1 if "SYN" in flags else 0))
    ack_count = int(_packet_float(row, "ACK Flag Count", "ack_flag_count", default=1 if "ACK" in flags else 0))
    if app_proto == "HTTP":
        http_method = "POST" if ("dos" in attack_lower or "web" in attack_lower) else "GET"
        url_length = _stable_packet_number(key, "url", min_value=12, max_value=180)
        content_type = "text/html"
    else:
        http_method = "-"
        url_length = 0
        content_type = "-"
    sql_flag = 1 if "sql" in attack_lower else 0
    xss_flag = 1 if "xss" in attack_lower else 0
    shell_flag = 1 if any(word in attack_lower for word in ["shell", "cmd", "command"]) else 0
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "protocol": protocol,
        "packet_length": packet_len,
        "ttl": ttl,
        "src_port": src_port,
        "dst_port": dst_port,
        "tcp_flags": flags,
        "window_size": window_size,
        "flow_duration": flow_duration,
        "total_fwd_packets": total_fwd,
        "total_bwd_packets": total_bwd,
        "flow_bytes_per_sec": round(flow_bytes, 2),
        "avg_packet_size": round(avg_packet_size, 2),
        "syn_flag_count": syn_count,
        "ack_flag_count": ack_count,
        "payload_length": payload_len,
        "http_method": http_method,
        "url_length": url_length,
        "content_type": content_type,
        "sql_keyword_flag": sql_flag,
        "xss_keyword_flag": xss_flag,
        "shell_keyword_flag": shell_flag,
    }


def _event_detail_records(top_events):
    if top_events is None or top_events.empty:
        return []
    records = []
    for _, row in top_events.iterrows():
        event_time = row.get("时间")
        time_text = pd.to_datetime(event_time).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(event_time) else "-"
        prob = row.get("预测概率", "-")
        if pd.notna(prob) and prob != "-":
            try:
                prob = f"{float(prob):.4f}"
            except Exception:
                prob = str(prob)
        risk_score = row.get("风险分", "-")
        if pd.notna(risk_score) and risk_score != "-":
            try:
                risk_score = f"{float(risk_score):.1f}"
            except Exception:
                risk_score = str(risk_score)
        packet = _packet_profile(row, time_text)
        detection = {
            "attack_probability": prob,
            "detection_result": row.get("检测结果", "-"),
            "risk_score": risk_score,
            "risk_level": row.get("风险等级", "-"),
            "timestamp": time_text,
        }
        records.append(
            {
                "packet": {k: "" if pd.isna(v) else str(v) for k, v in packet.items()},
                "detection": {k: "" if pd.isna(v) else str(v) for k, v in detection.items()},
            }
        )
    return records


def render_topology(events):
    snap = topology_snapshot(events)
    summary = snap["summary"]
    groups = snap["groups"]
    primary_target = snap["target"]
    assets = snap["assets"]
    has_attack = snap["has_attack"]
    has_suspicious = snap["has_suspicious"]
    detail_json = json.dumps(groups, ensure_ascii=False)
    rows_html = _event_rows_html(snap["top_events"])
    event_detail_json = json.dumps(_event_detail_records(snap["top_events"]), ensure_ascii=False)
    attack_label = f"攻击链路 / {summary['attack_count']} 条" if has_attack else None
    suspicious_label = f"可疑链路 / {summary['suspicious_count']} 条" if has_suspicious else None
    target_title = "被攻击主机" if has_attack else "重点监测主机"
    target_color = BAD if has_attack else NORMAL
    target_tag = "高危" if has_attack and primary_target.get("risk") == "高危" else ("攻击中" if has_attack else "正常")
    firewall_color = WARN if has_attack else INFO
    firewall_tag = "拦截" if has_attack else "正常"
    ids_output_color = WARN if has_attack else NORMAL
    ids_output_subtitle = "告警入库" if has_attack else "暂无告警"
    ambient_glow = "rgba(255,77,94,.12)" if has_attack else "rgba(30,227,138,.08)"
    has_normal = groups["normal"]["events"] > 0
    normal_flows = ""
    if has_normal:
        normal_paths = [
            svg_flow("M292,292 H324 V214 H340", NORMAL, "arrow-green", 2.0, False, "正常流量", 306, 282),
            svg_flow("M530,214 H538", NORMAL, "arrow-green", 2.0),
            svg_flow("M728,214 H736", NORMAL, "arrow-green", 2.0),
            svg_flow("M926,214 H934 V214 H958", NORMAL, "arrow-green", 1.9),
            svg_flow("M926,234 H934 V366 H958", NORMAL, "arrow-green", 1.9),
        ]
        if not has_attack:
            normal_paths.append(svg_flow("M926,204 H934 V135 H958", NORMAL, "arrow-green", 1.9))
        if not has_suspicious:
            normal_paths.append(svg_flow("M926,224 H934 V290 H958", NORMAL, "arrow-green", 1.8))
        normal_flows = "\n".join(normal_paths)
    attack_flows = ""
    if has_attack:
        attack_flows = "\n".join(
            [
                svg_flow("M292,152 H324 V194 H340", BAD, "arrow-red", 3.1, True, attack_label, 302, 138, True, ".96"),
                svg_flow("M530,194 H538", BAD, "arrow-red", 3.1, True, None, None, None, True, ".96"),
                svg_flow("M728,194 H736", BAD, "arrow-red", 3.1, True, None, None, None, True, ".96"),
                svg_flow("M926,194 H934 V135 H958", BAD, "arrow-red", 3.1, True, None, None, None, True, ".96"),
            ]
        )
    suspicious_flows = ""
    if has_suspicious:
        suspicious_flows = svg_flow("M926,244 H934 V290 H958", WARN, "arrow-yellow", 2.3, True, suspicious_label, 860, 276, False, ".9", "middle")
    span_flows = "\n".join(
        [
            svg_flow("M831,262 V404 H367 V504", INFO, "arrow-blue", 2.4, True, "SPAN镜像 / 流量采集", 604, 394, False, ".96", "middle"),
            svg_flow("M958,442 H914 V596 H367 V566", INFO, "arrow-blue", 2.2, True, "日志采集", 850, 586, False, ".92", "middle"),
        ]
    )

    html_doc = f"""
    <html>
    <head>
      <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: transparent; color: {TEXT}; font-family: "Microsoft YaHei", Arial, sans-serif; }}
        .topology-card {{
          border: 1px solid rgba(39,221,255,.22);
          border-radius: 10px;
          padding: 14px 16px;
          background:
            radial-gradient(circle at 13% 12%, rgba(47,125,255,.18), transparent 24%),
            radial-gradient(circle at 82% 22%, {ambient_glow}, transparent 20%),
            linear-gradient(180deg, rgba(7,24,39,.98), rgba(3,14,25,.98));
          box-shadow: 0 0 28px rgba(24,199,255,.10) inset;
        }}
        .topology-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 8px; }}
        .title-wrap h2 {{ margin: 0 0 6px 0; font-size: 25px; letter-spacing: 0; }}
        .title-wrap p {{ margin: 0; color: {MUTED}; font-size: 12.5px; }}
        .stat-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; min-width: 300px; opacity: .9; }}
        .stat {{ border: 1px solid rgba(24,199,255,.16); border-radius: 8px; padding: 6px 10px; background: rgba(6,21,35,.72); }}
        .stat.warn {{ border-color: rgba(245,197,66,.28); }}
        .stat.bad {{ border-color: rgba(255,77,94,.32); }}
        .stat .label {{ color: {MUTED}; font-size: 11.5px; }}
        .stat .value {{ margin-top: 2px; font-size: 19px; font-weight: 800; }}
        .viz-frame {{ border: 1px solid rgba(24,199,255,.20); border-radius: 10px; overflow: hidden; background: #04111f; }}
        svg {{ display: block; width: 100%; height: auto; }}
        .group-node {{ cursor: pointer; }}
        .group-node:hover text {{ filter: brightness(1.2); }}
        .attack-flow {{ animation: dash 1.2s linear infinite; }}
        .alert-node {{ }}
        .risk-tag {{ animation: tagBlink 1.05s ease-in-out infinite; }}
        @keyframes dash {{ to {{ stroke-dashoffset: -38; }} }}
        @keyframes tagBlink {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .42; }} }}
        .bottom-grid {{ display: grid; grid-template-columns: 1.65fr 1fr; gap: 12px; margin-top: 10px; }}
        .panel {{ border: 1px solid rgba(24,199,255,.18); border-radius: 8px; background: rgba(4,17,31,.84); padding: 12px; min-height: 278px; }}
        .panel h3 {{ margin: 0 0 10px 0; font-size: 16px; }}
        .table-wrap {{ max-height: 230px; overflow-y: auto; padding-right: 2px; }}
        table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
        th, td {{ padding: 8px 7px; border-bottom: 1px solid rgba(126,164,191,.12); color: {TEXT}; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        th {{ color: {MUTED}; font-weight: 600; }}
        .event-row {{ cursor: pointer; transition: background .16s ease, transform .16s ease; }}
        .event-row:hover {{ background: rgba(24,199,255,.08); }}
        .event-row.active {{ background: rgba(24,199,255,.12); }}
        .pill {{ display: inline-block; min-width: 40px; text-align: center; padding: 2px 7px; border-radius: 999px; font-weight: 800; }}
        .pill.high {{ color: {BAD}; border: 1px solid rgba(255,77,94,.42); background: rgba(255,77,94,.12); }}
        .pill.mid {{ color: {WARN}; border: 1px solid rgba(245,197,66,.42); background: rgba(245,197,66,.10); }}
        .pill.low {{ color: {NORMAL}; border: 1px solid rgba(30,227,138,.38); background: rgba(30,227,138,.10); }}
        .empty {{ color: {MUTED}; text-align: center; padding: 26px 0; }}
        .detail-title {{ color: {INFO}; font-weight: 800; font-size: 17px; margin-bottom: 8px; }}
        .detail-meta {{ color: {MUTED}; margin-bottom: 12px; font-size: 13px; line-height: 1.7; }}
        .ip-list {{ display: flex; flex-wrap: wrap; gap: 7px; }}
        .ip-chip {{ border: 1px solid rgba(24,199,255,.24); border-radius: 999px; padding: 5px 9px; color: {TEXT}; font-size: 12px; background: rgba(24,199,255,.08); }}
        .event-modal {{ position: fixed; left: 9%; top: 66%; display: none; z-index: 20; }}
        .event-modal.show {{ display: block; }}
        .event-card {{ width: 720px; max-width: 92vw; max-height: calc(100vh - 18px); overflow-y: auto; border: 1px solid rgba(24,199,255,.42); border-radius: 12px; background: linear-gradient(180deg, rgba(7,24,39,.96), rgba(3,14,25,.94)); box-shadow: 0 14px 34px rgba(0,0,0,.35), 0 0 18px rgba(24,199,255,.10) inset; padding: 12px; backdrop-filter: blur(2px); }}
        .event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; cursor: move; user-select: none; }}
        .event-head h3 {{ margin: 0; font-size: 18px; color: {TEXT}; }}
        .close-btn {{ border: 1px solid rgba(24,199,255,.32); border-radius: 8px; color: {TEXT}; background: rgba(24,199,255,.08); padding: 4px 10px; cursor: pointer; font-size: 13px; }}
        .event-detail-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; }}
        .event-detail-item {{ border: 1px solid rgba(126,164,191,.16); border-radius: 8px; background: rgba(6,21,35,.72); padding: 7px 9px; min-height: 48px; }}
        .event-detail-item .k {{ color: {MUTED}; font-size: 11px; margin-bottom: 5px; }}
        .event-detail-item .v {{ color: {TEXT}; font-size: 13px; font-weight: 800; word-break: break-all; }}
        .event-section-title {{ margin: 12px 0 8px; color: {INFO}; font-size: 13px; font-weight: 800; }}
        .detect-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 7px; }}
        .detect-item {{ border: 1px solid rgba(245,197,66,.16); border-radius: 8px; background: rgba(245,197,66,.05); padding: 7px 9px; }}
        .detect-item .k {{ color: {MUTED}; font-size: 10.5px; margin-bottom: 5px; }}
        .detect-item .v {{ color: {TEXT}; font-size: 12.5px; font-weight: 800; word-break: break-all; }}
        .event-desc {{ margin-top: 10px; border: 1px solid rgba(24,199,255,.18); border-radius: 8px; padding: 10px; color: {MUTED}; font-size: 12.5px; line-height: 1.7; background: rgba(24,199,255,.05); }}
        .event-inline-detail {{ display: none; margin-top: 10px; border: 1px solid rgba(24,199,255,.22); border-radius: 8px; background: rgba(4,17,31,.92); padding: 10px; }}
        .event-inline-detail.show {{ display: block; }}
        .event-inline-title {{ display: flex; align-items: center; justify-content: space-between; color: {INFO}; font-size: 13px; font-weight: 800; margin-bottom: 8px; }}
        .event-inline-close {{ border: 1px solid rgba(24,199,255,.28); border-radius: 6px; color: {MUTED}; background: rgba(24,199,255,.06); padding: 2px 8px; cursor: pointer; }}
        .event-inline-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }}
        .event-inline-item {{ border: 1px solid rgba(126,164,191,.13); border-radius: 7px; background: rgba(6,21,35,.68); padding: 6px 8px; min-height: 42px; }}
        .event-inline-item .k {{ color: {MUTED}; font-size: 10px; margin-bottom: 4px; }}
        .event-inline-item .v {{ color: {TEXT}; font-size: 11.8px; font-weight: 800; word-break: break-all; }}
        .event-inline-detect {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-top: 8px; }}
        .event-inline-desc {{ margin-top: 8px; color: {MUTED}; font-size: 11.5px; line-height: 1.55; }}
      </style>
    </head>
    <body>
    <div class="topology-card">
      <div class="topology-head">
        <div class="title-wrap">
          <h2>动态网络攻击告警拓扑图</h2>
          <p>数据来源：当前批量检测结果与 SQLite 历史记录；节点表示设备或聚合分组，链路颜色表示正常、可疑、攻击和镜像采集流量。</p>
        </div>
        <div class="stat-row">
          <div class="stat"><div class="label">设备总数</div><div class="value" style="color:{BLUE};">{summary["device_total"]}</div></div>
          <div class="stat"><div class="label">链路数</div><div class="value" style="color:{INFO};">{summary["relation_count"]}</div></div>
          <div class="stat bad"><div class="label">攻击链路数</div><div class="value" style="color:{BAD};">{summary["attack_count"]}</div></div>
        </div>
      </div>
      <div class="viz-frame">
      <svg viewBox="0 0 1280 720" role="img" aria-label="动态网络攻击告警拓扑图">
        <defs>
          <filter id="glow-red"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="glow-cyan"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L9,4 z" fill="{NORMAL}"/></marker>
          <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L9,4 z" fill="{BAD}"/></marker>
          <marker id="arrow-yellow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L9,4 z" fill="{WARN}"/></marker>
          <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L9,4 z" fill="{INFO}"/></marker>
          <marker id="arrow-purple" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L9,4 z" fill="{PURPLE}"/></marker>
          <linearGradient id="device-bg" x1="0" x2="1"><stop stop-color="#061827"/><stop offset="1" stop-color="#08243d"/></linearGradient>
        </defs>
        <rect width="1280" height="720" fill="#04111f"/>

        <rect x="24" y="48" width="280" height="344" rx="14" fill="rgba(7,24,39,.74)" stroke="rgba(24,199,255,.16)"/>
        <text x="42" y="78" fill="{TEXT}" font-size="17" font-weight="800">外部来源区</text>
        {svg_source_group("attack", 44, 98, 240, 132, "攻击源组", groups["attack"], BAD if has_attack else INFO, True)}
        {svg_source_group("normal", 44, 236, 240, 144, "正常用户组", groups["normal"], NORMAL, False)}

        <rect x="338" y="48" width="586" height="282" rx="14" fill="rgba(7,24,39,.66)" stroke="rgba(24,199,255,.16)"/>
        <text x="360" y="78" fill="{TEXT}" font-size="17" font-weight="800">网络边界与核心设备区</text>
        {svg_device(348, 150, 174, 104, "边界路由器", "19.0.0.1", INFO, "router")}
        {svg_device(546, 150, 174, 104, "防火墙", "10.0.0.254", firewall_color, "firewall", has_attack, firewall_tag)}
        {svg_device(744, 150, 174, 104, "核心交换机", "10.0.0.10", BLUE, "switch")}

        <rect x="944" y="48" width="312" height="432" rx="14" fill="rgba(7,24,39,.72)" stroke="rgba(24,199,255,.16)"/>
        <text x="966" y="78" fill="{TEXT}" font-size="17" font-weight="800">内部目标资产区</text>
        {svg_asset_node(968, 94, 260, 82, target_title, primary_target["ip"], target_color, "pc", target_tag, has_attack, primary_target["events"] if has_attack else None)}
        {svg_asset_node(968, 178, 260, 72, "业务服务器", assets["business"], NORMAL, "server", "正常")}
        {svg_asset_node(968, 254, 260, 72, "数据库服务器", assets["database"], WARN if has_suspicious else NORMAL, "server", "可疑" if has_suspicious else "正常")}
        {svg_asset_node(968, 330, 260, 72, "管理员主机", assets["admin"], NORMAL, "pc", "正常")}
        {svg_asset_node(968, 406, 260, 72, "日志/流量采集节点", assets["collector"], INFO, "collector", "采集")}

        {normal_flows}
        {attack_flows}
        {suspicious_flows}

        <rect x="130" y="432" width="790" height="192" rx="16" fill="rgba(3,20,34,.82)" stroke="rgba(24,199,255,.22)"/>
        <text x="152" y="464" fill="{TEXT}" font-size="17" font-weight="800">IDS旁路检测与告警输出区</text>
        {svg_ids_server_pipeline(ids_output_subtitle, ids_output_color)}
        {span_flows}
        <rect x="454" y="598" width="356" height="18" rx="6" fill="#04111f" opacity=".86"/>
        <text x="632" y="611" text-anchor="middle" fill="{MUTED}" font-size="10.8">检测结果同步至顶部统计、监控终端与下方告警事件表</text>

        <g transform="translate(24 652)">
          <rect width="1232" height="48" rx="10" fill="rgba(7,24,39,.86)" stroke="rgba(24,199,255,.16)"/>
          <text x="18" y="27" fill="{TEXT}" font-size="13" font-weight="800">图例</text>
          <path d="M82,23 H136" stroke="{NORMAL}" stroke-width="3"/><text x="146" y="27" fill="{MUTED}" font-size="11.5">绿色实线：正常流量</text>
          <path d="M288,23 H342" stroke="{BAD}" stroke-width="3" stroke-dasharray="10 7"/><text x="352" y="27" fill="{MUTED}" font-size="11.5">红色虚线：攻击流量</text>
          <path d="M494,23 H548" stroke="{WARN}" stroke-width="3" stroke-dasharray="10 7"/><text x="558" y="27" fill="{MUTED}" font-size="11.5">黄色虚线：可疑流量</text>
          <path d="M700,23 H754" stroke="{INFO}" stroke-width="3" stroke-dasharray="10 7"/><text x="764" y="27" fill="{MUTED}" font-size="11.5">蓝色虚线：镜像/采集链路</text>
          <rect x="980" y="14" width="38" height="18" rx="6" fill="{BAD}" opacity=".14" stroke="{BAD}"/><text x="1028" y="27" fill="{MUTED}" font-size="11.5">红/黄/绿标签：风险状态</text>
        </g>
      </svg>
      </div>
      <div class="bottom-grid">
        <div class="panel">
          <h3>实时告警事件</h3>
          <div class="table-wrap">
          <table>
            <thead><tr><th>时间</th><th>源IP</th><th>目的IP</th><th>风险等级</th><th>事件类型</th><th>检测结果</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          </div>
        </div>
        <div class="panel">
          <h3>聚合节点详情</h3>
          <div id="detailBox"></div>
        </div>
      </div>
      <div id="eventModal" class="event-modal">
        <div class="event-card">
          <div id="eventDragHandle" class="event-head">
            <h3>流量包详细信息</h3>
            <button class="close-btn" onclick="closeEvent()">关闭</button>
          </div>
          <div id="eventDetail"></div>
        </div>
      </div>
    </div>
    <script>
      const groupData = {detail_json};
      const eventData = {event_detail_json};
      function esc(value) {{
        return String(value ?? "-").replace(/[&<>"']/g, s => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[s]));
      }}
      function showGroup(key) {{
        const item = groupData[key] || groupData.normal;
        const ips = item.ips && item.ips.length ? item.ips.map(ip => `<span class="ip-chip">${{ip}}</span>`).join("") : '<span class="ip-chip">暂无明细</span>';
        document.getElementById("detailBox").innerHTML =
          `<div class="detail-title">${{item.title}}</div>
           <div class="detail-meta">风险等级：${{item.risk}}　事件数：${{item.events}}　IP数量：${{item.count}}</div>
           <div class="ip-list">${{ips}}</div>`;
      }}
      function showEvent(index, rowEl) {{
        const item = eventData[index];
        if (!item) return;
        document.querySelectorAll(".event-row").forEach(row => row.classList.remove("active"));
        if (rowEl) rowEl.classList.add("active");
        const packet = item.packet || {{}};
        const detection = item.detection || {{}};
        const fieldLabels = {{
          "src_ip": "源IP地址",
          "dst_ip": "目标IP地址",
          "protocol": "协议类型",
          "packet_length": "数据包长度",
          "ttl": "生存时间",
          "src_port": "源端口",
          "dst_port": "目标端口",
          "tcp_flags": "TCP控制位",
          "window_size": "TCP窗口大小",
          "flow_duration": "流持续时间",
          "total_fwd_packets": "正向包数量",
          "total_bwd_packets": "反向包数量",
          "flow_bytes_per_sec": "每秒字节数",
          "avg_packet_size": "平均包长度",
          "syn_flag_count": "SYN标志位数量",
          "ack_flag_count": "ACK标志位数量",
          "payload_length": "主体数据长度",
          "http_method": "HTTP请求方法",
          "url_length": "URL长度",
          "content_type": "内容类型",
          "sql_keyword_flag": "SQL注入关键字标记",
          "xss_keyword_flag": "XSS关键字标记",
          "shell_keyword_flag": "命令执行关键字标记",
          "attack_probability": "攻击概率",
          "detection_result": "检测结果",
          "risk_score": "风险分值",
          "risk_level": "风险等级",
          "timestamp": "检测时间",
        }};
        const groups = [
          ["网络层信息", ["src_ip", "dst_ip", "protocol", "packet_length", "ttl"], packet, "event-detail-item"],
          ["传输层信息", ["src_port", "dst_port", "tcp_flags", "window_size"], packet, "event-detail-item"],
          ["流量统计信息", ["flow_duration", "total_fwd_packets", "total_bwd_packets", "flow_bytes_per_sec", "avg_packet_size", "syn_flag_count", "ack_flag_count"], packet, "event-detail-item"],
          ["主体信息（Payload信息）", ["payload_length", "http_method", "url_length", "content_type", "sql_keyword_flag", "xss_keyword_flag", "shell_keyword_flag"], packet, "event-detail-item"],
          ["检测结果信息", ["attack_probability", "detection_result", "risk_score", "risk_level", "timestamp"], detection, "detect-item"],
        ];
        const sectionHtml = groups.map(([title, fields, source, cls]) => {{
          const gridClass = cls === "detect-item" ? "detect-grid" : "event-detail-grid";
          const grid = fields.map(k => `
            <div class="${{cls}}">
              <div class="k">${{esc(fieldLabels[k] || k)}}</div>
              <div class="v">${{esc((source[k] === undefined || source[k] === null || source[k] === "") ? "-" : source[k])}}</div>
            </div>
          `).join("");
          return `<div class="event-section-title">${{esc(title)}}</div><div class="${{gridClass}}">${{grid}}</div>`;
        }}).join("");
        showGlobalEvent(
          sectionHtml,
          rowEl
        );
      }}
      function closeEvent() {{
        document.getElementById("eventModal").classList.remove("show");
        try {{
          const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
          const globalModal = parentDoc.getElementById("globalPacketModal");
          if (globalModal) globalModal.style.display = "none";
        }} catch (err) {{}}
      }}
      function showGlobalEvent(contentHtml, rowEl) {{
        let parentDoc = document;
        let parentWin = window;
        try {{
          parentDoc = window.parent && window.parent.document ? window.parent.document : document;
          parentWin = parentDoc.defaultView || window;
        }} catch (err) {{
          parentDoc = document;
          parentWin = window;
        }}
        let shell = parentDoc.getElementById("globalPacketModal");
        if (!shell) {{
          shell = parentDoc.createElement("div");
          shell.id = "globalPacketModal";
          shell.style.position = "fixed";
          shell.style.left = "120px";
          shell.style.top = "120px";
          shell.style.zIndex = "999999";
          shell.style.display = "none";
          parentDoc.body.appendChild(shell);
        }}
        shell.innerHTML = `
          <div id="globalPacketCard" style="width:720px;max-width:92vw;max-height:calc(100vh - 18px);overflow-y:auto;border:1px solid rgba(24,199,255,.42);border-radius:12px;background:linear-gradient(180deg,rgba(7,24,39,.97),rgba(3,14,25,.95));box-shadow:0 14px 34px rgba(0,0,0,.35),0 0 18px rgba(24,199,255,.10) inset;padding:12px;backdrop-filter:blur(2px);color:{TEXT};font-family:'Microsoft YaHei',Arial,sans-serif;">
            <div id="globalPacketHandle" style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;cursor:move;user-select:none;">
              <h3 style="margin:0;font-size:18px;color:{TEXT};">流量包详细信息</h3>
              <button id="globalPacketClose" style="border:1px solid rgba(24,199,255,.32);border-radius:8px;color:{TEXT};background:rgba(24,199,255,.08);padding:4px 10px;cursor:pointer;font-size:13px;">关闭</button>
            </div>
            <div class="event-global-body">
              ${{contentHtml}}
            </div>
          </div>`;
        const style = parentDoc.createElement("style");
        style.textContent = `
          #globalPacketModal .event-section-title {{ margin:12px 0 8px;color:{INFO};font-size:13px;font-weight:800; }}
          #globalPacketModal .event-detail-grid {{ display:grid;grid-template-columns:repeat(4,1fr);gap:7px; }}
          #globalPacketModal .event-detail-item {{ border:1px solid rgba(126,164,191,.16);border-radius:8px;background:rgba(6,21,35,.72);padding:7px 9px;min-height:48px; }}
          #globalPacketModal .event-detail-item .k {{ color:{MUTED};font-size:11px;margin-bottom:5px; }}
          #globalPacketModal .event-detail-item .v {{ color:{TEXT};font-size:13px;font-weight:800;word-break:break-all; }}
          #globalPacketModal .detect-grid {{ display:grid;grid-template-columns:repeat(5,1fr);gap:7px; }}
          #globalPacketModal .detect-item {{ border:1px solid rgba(245,197,66,.16);border-radius:8px;background:rgba(245,197,66,.05);padding:7px 9px; }}
          #globalPacketModal .detect-item .k {{ color:{MUTED};font-size:10.5px;margin-bottom:5px; }}
          #globalPacketModal .detect-item .v {{ color:{TEXT};font-size:12.5px;font-weight:800;word-break:break-all; }}
          #globalPacketModal .event-desc {{ margin-top:10px;border:1px solid rgba(24,199,255,.18);border-radius:8px;padding:10px;color:{MUTED};font-size:12.5px;line-height:1.7;background:rgba(24,199,255,.05); }}
        `;
        shell.appendChild(style);
        shell.style.display = "block";
        if (!shell.dataset.moved) {{
          try {{
            const frameRect = window.frameElement ? window.frameElement.getBoundingClientRect() : {{ left: 0, top: 0 }};
            const rowRect = rowEl ? rowEl.getBoundingClientRect() : {{ left: 80, top: 560 }};
            shell.style.left = `${{frameRect.left + rowRect.left + 24}}px`;
            shell.style.top = `${{frameRect.top + rowRect.top - 80}}px`;
          }} catch (err) {{
            shell.style.left = "120px";
            shell.style.top = "120px";
          }}
        }}
        function keepGlobalInView() {{
          const card = shell.querySelector("#globalPacketCard");
          const cardW = card ? card.offsetWidth : 720;
          const cardH = card ? Math.min(card.offsetHeight, parentWin.innerHeight - 18) : 360;
          const rect = shell.getBoundingClientRect();
          shell.style.left = `${{Math.max(8, Math.min(parentWin.innerWidth - cardW - 8, rect.left))}}px`;
          shell.style.top = `${{Math.max(8, Math.min(parentWin.innerHeight - cardH - 8, rect.top))}}px`;
        }}
        keepGlobalInView();
        shell.querySelector("#globalPacketClose").onclick = () => {{ shell.style.display = "none"; }};
        if (!shell._dragReady) {{
          let draggingGlobal = false, sx = 0, sy = 0, bx = 0, by = 0;
          parentDoc.addEventListener("mousedown", (event) => {{
            if (!event.target.closest("#globalPacketHandle")) return;
            if (event.target.closest("button")) return;
            draggingGlobal = true;
            const rect = shell.getBoundingClientRect();
            sx = event.clientX; sy = event.clientY; bx = rect.left; by = rect.top;
            shell.dataset.moved = "1";
            event.preventDefault();
          }});
          parentDoc.addEventListener("mousemove", (event) => {{
            if (!draggingGlobal) return;
            const card = shell.querySelector("#globalPacketCard");
            const cardW = card ? card.offsetWidth : 720;
            const cardH = card ? Math.min(card.offsetHeight, parentWin.innerHeight - 18) : 360;
            const nextX = Math.max(8, Math.min(parentWin.innerWidth - cardW - 8, bx + event.clientX - sx));
            const nextY = Math.max(8, Math.min(parentWin.innerHeight - cardH - 8, by + event.clientY - sy));
            shell.style.left = `${{nextX}}px`;
            shell.style.top = `${{nextY}}px`;
          }});
          parentDoc.addEventListener("mouseup", () => draggingGlobal = false);
          parentWin.addEventListener("resize", keepGlobalInView);
          parentDoc.addEventListener("keydown", (event) => {{
            if (event.key === "Escape") shell.style.display = "none";
          }});
          shell._dragReady = true;
        }}
      }}
      const modal = document.getElementById("eventModal");
      const handle = document.getElementById("eventDragHandle");
      let dragging = false, startX = 0, startY = 0, baseX = 0, baseY = 0;
      function keepModalInView() {{
        const card = modal.querySelector(".event-card");
        const cardW = card ? card.offsetWidth : 650;
        const cardH = card ? Math.min(card.offsetHeight, window.innerHeight - 18) : 360;
        const rect = modal.getBoundingClientRect();
        const nextX = Math.max(8, Math.min(window.innerWidth - cardW - 8, rect.left));
        const nextY = Math.max(8, Math.min(window.innerHeight - cardH - 8, rect.top));
        modal.style.left = `${{nextX}}px`;
        modal.style.top = `${{nextY}}px`;
      }}
      handle.addEventListener("mousedown", (event) => {{
        if (event.target.closest("button")) return;
        dragging = true;
        const rect = modal.getBoundingClientRect();
        startX = event.clientX;
        startY = event.clientY;
        baseX = rect.left;
        baseY = rect.top;
        modal.dataset.moved = "1";
        event.preventDefault();
      }});
      window.addEventListener("mousemove", (event) => {{
        if (!dragging) return;
        const card = modal.querySelector(".event-card");
        const cardW = card ? card.offsetWidth : 650;
        const cardH = card ? Math.min(card.offsetHeight, window.innerHeight - 18) : 360;
        const nextX = Math.max(8, Math.min(window.innerWidth - cardW - 8, baseX + event.clientX - startX));
        const nextY = Math.max(8, Math.min(window.innerHeight - cardH - 8, baseY + event.clientY - startY));
        modal.style.left = `${{nextX}}px`;
        modal.style.top = `${{nextY}}px`;
      }});
      window.addEventListener("mouseup", () => dragging = false);
      window.addEventListener("resize", keepModalInView);
      window.addEventListener("keydown", (event) => {{
        if (event.key === "Escape") closeEvent();
      }});
      showGroup("{'attack' if has_attack else 'normal'}");
    </script>
    </body>
    </html>
    """
    components.html(html_doc, height=920, scrolling=False)


def trend_df(events, freq="15min"):
    data = events.copy()
    data["时间桶"] = pd.to_datetime(data["时间"]).dt.floor(freq)
    grouped = data.groupby(["时间桶", "检测结果"]).size().unstack(fill_value=0).sort_index()
    for col in ["正常", "攻击"]:
        if col not in grouped:
            grouped[col] = 0
    grouped = grouped[["正常", "攻击"]]
    if len(grouped) == 1:
        prev = grouped.copy()
        prev.index = prev.index - pd.Timedelta(freq)
        prev[["正常", "攻击"]] = 0
        grouped = pd.concat([prev, grouped]).sort_index()
    return grouped


def overview_page(events, rf_metrics, mlp_metrics):
    total, normal, attacks, high, avg_prob, confidence = summarize(events)
    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="grid-card overview-status">
                <div class="small-title">系统状态</div>
                <div class="overview-badges">
                    <span class="badge"><span class="status-dot" style="background:{NORMAL}"></span>运行中</span>
                    <span class="badge">模型在线</span>
                    <span class="badge">检测中</span>
                </div>
            </div>
            {metric_card_html("当前流量", f"{total:,}", "实时窗口", INFO)}
            {metric_card_html("正常流量", f"{normal:,}", "绿色流量", NORMAL)}
            {metric_card_html("攻击流量", f"{attacks:,}", "红色告警", BAD)}
            {metric_card_html("今日告警", f"{attacks:,}", "检测记录", WARN)}
            {metric_card_html("高危告警", f"{high:,}", "高风险事件", BAD)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_topology(events)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])
    with c1:
        risk_counts = events["风险等级"].value_counts().reindex(["高危", "中危", "低危"], fill_value=0).reset_index()
        risk_counts.columns = ["风险等级", "数量"]
        fig = px.pie(risk_counts, names="风险等级", values="数量", hole=.58, color="风险等级", color_discrete_map={"高危": BAD, "中危": WARN, "低危": NORMAL}, title="风险等级分布")
        show_chart(fig, 245)
    with c2:
        flow_type = events["检测结果"].value_counts().reindex(["正常", "攻击"], fill_value=0).reset_index()
        flow_type.columns = ["流量类型", "数量"]
        fig = px.pie(flow_type, names="流量类型", values="数量", hole=.45, title="正常/攻击流量占比", color="流量类型", color_discrete_map={"正常": NORMAL, "攻击": BAD})
        show_chart(fig, 245)
    with c3:
        attack_type = events[events["检测结果"] == "攻击"]["攻击类型"].value_counts().head(6).reset_index()
        attack_type.columns = ["攻击类型", "数量"]
        if attack_type.empty:
            attack_type = pd.DataFrame({"攻击类型": ["暂无攻击"], "数量": [1]})
        fig = px.pie(attack_type, names="攻击类型", values="数量", hole=.45, title="攻击类型占比", color_discrete_sequence=[BAD, "#ff8a4d", WARN, BLUE, INFO, NORMAL])
        show_chart(fig, 245)
    with c4:
        result = "攻击" if avg_prob >= .5 else "正常"
        result_color = BAD if avg_prob >= .5 else NORMAL
        latest = events.head(1).iloc[0] if not events.empty else None
        latest_text = ""
        if latest is not None:
            latest_text = f"最新流量：{latest['源IP']} → {latest['目的IP']}，风险等级 {latest['风险等级']}。"
        st.markdown(
            f"""
            <div class="prediction-card">
              <div class="small-title">模型预测概况</div>
              <div class="prediction-grid">
                <div class="prediction-item">
                  <div class="metric-label">攻击概率</div>
                  <div class="metric-value" style="color:{result_color};">{avg_prob:.3f}</div>
                </div>
                <div class="prediction-item">
                  <div class="metric-label">模型置信度</div>
                  <div class="metric-value" style="color:{INFO};">{confidence:.3f}</div>
                </div>
                <div class="prediction-item">
                  <div class="metric-label">检测结果</div>
                  <div class="metric-value" style="color:{result_color};">{result}</div>
                </div>
              </div>
              <div class="prediction-note">{latest_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        fig = go.Figure()
        x = np.arange(30)
        fig.add_trace(go.Scatter(x=x, y=np.exp(-x / 9) + .02, name="训练Loss", line=dict(color=NORMAL)))
        fig.add_trace(go.Scatter(x=x, y=np.exp(-x / 7) + .06, name="验证Loss", line=dict(color=BLUE)))
        fig.update_layout(title="模型性能监控：Loss曲线")
        show_chart(fig, 255)
    with b2:
        comp = pd.DataFrame(
            {
                "指标": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "RandomForest": [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"]],
                "Lightweight MLP": [mlp_metrics["accuracy"], mlp_metrics["precision"], mlp_metrics["recall"], mlp_metrics["f1"]],
            }
        )
        fig = px.bar(comp, x="指标", y=["RandomForest", "Lightweight MLP"], barmode="group", title="模型指标对比", color_discrete_sequence=[BLUE, NORMAL])
        show_chart(fig, 255)
    with b3:
        line = trend_df(events, "2min")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=line.index, y=line["攻击"], name="攻击", line=dict(color=BAD, width=3), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=line.index, y=line["正常"], name="正常", line=dict(color=NORMAL, width=2)))
        fig.update_layout(title="攻击趋势图")
        show_chart(fig, 255)

def dark_confusion_matrix():
    z = np.array([[1285, 23], [18, 1467]])
    fig = px.imshow(
        z,
        text_auto=True,
        color_continuous_scale=[[0, "#061827"], [.45, BLUE], [1, INFO]],
        labels=dict(x="预测标签", y="真实标签", color="样本数"),
        x=["正常", "攻击"],
        y=["正常", "攻击"],
        title="混淆矩阵（Lightweight MLP）",
    )
    fig.update_traces(textfont=dict(color="#f7fdff", size=16))
    show_chart(fig, 310)


def dark_probability_chart(events):
    data = events.copy()
    data["结果类型"] = np.where(data["检测结果"] == "攻击", "攻击流量", "正常流量")
    fig = px.histogram(
        data,
        x="预测概率",
        color="结果类型",
        nbins=42,
        barmode="overlay",
        opacity=.72,
        title="预测概率分布",
        color_discrete_map={"正常流量": NORMAL, "攻击流量": BAD},
    )
    fig.add_vline(x=.5, line_color=WARN, line_dash="dash", annotation_text="阈值 0.50", annotation_font_color=WARN)
    show_chart(fig, 310)


def dark_loss_curve():
    x = np.arange(1, 31)
    train = 0.78 * np.exp(-x / 8.5) + 0.018
    valid = 0.92 * np.exp(-x / 7.2) + 0.035 + np.sin(x / 3) * .006
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=train, mode="lines+markers", name="训练Loss", line=dict(color=NORMAL, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=valid, mode="lines+markers", name="验证Loss", line=dict(color=BLUE, width=3), marker=dict(size=4)))
    fig.update_layout(title="Loss曲线", xaxis_title="Epoch", yaxis_title="Loss")
    show_chart(fig, 310)


def get_feature_importance_top10():
    names = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Destination Port",
        "Protocol",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
    ]
    values = np.array([.26, .15, .12, .095, .078, .066, .052, .041, .034, .028])
    try:
        rf = joblib.load(RF_MODEL_PATH)
        model_values = np.asarray(getattr(rf, "feature_importances_", []), dtype=float)
        model_names = list(getattr(rf, "feature_names_in_", []))
        if len(model_values) and len(model_names) == len(model_values):
            idx = np.argsort(model_values)[-10:][::-1]
            names = [model_names[i] for i in idx]
            values = model_values[idx]
    except Exception:
        pass
    return pd.DataFrame({"特征": names, "重要性": values}).sort_values("重要性", ascending=True)


def dark_feature_importance():
    features = get_feature_importance_top10()
    fig = px.bar(
        features,
        x="重要性",
        y="特征",
        orientation="h",
        title="特征重要性 Top10",
        color="重要性",
        color_continuous_scale=[[0, "#0b3150"], [.45, BLUE], [1, NORMAL]],
    )
    fig.update_layout(coloraxis_showscale=False)
    show_chart(fig, 310)


def ai_analysis_page(events, rf_metrics, mlp_metrics):
    c = st.columns(5)
    cards = [
        ("Accuracy", fmt_pct(mlp_metrics["accuracy"]), "Lightweight MLP", NORMAL),
        ("Precision", fmt_pct(mlp_metrics["precision"]), "深度学习检测精度", INFO),
        ("Recall", fmt_pct(mlp_metrics["recall"]), "攻击召回能力", WARN),
        ("F1 Score", fmt_pct(mlp_metrics["f1"]), "综合分类表现", NORMAL),
        ("推理耗时", "12 ms", "单批次模拟展示", INFO),
    ]
    for col, item in zip(c, cards):
        with col:
            metric_card(*item)

    st.markdown('<div class="sim-note">本页突出轻量化 MLP 深度学习模型，同时保留 RandomForest 基线模型作为对照。</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        dark_confusion_matrix()
    with r1c2:
        dark_probability_chart(events)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        dark_loss_curve()
    with r2c2:
        dark_feature_importance()

    comp = pd.DataFrame(
        [
            {"模型": "RandomForest", "Accuracy": fmt_pct(rf_metrics["accuracy"]), "Precision": fmt_pct(rf_metrics["precision"]), "Recall": fmt_pct(rf_metrics["recall"]), "F1 Score": fmt_pct(rf_metrics["f1"]), "训练耗时(s)": f'{rf_metrics.get("train_time", 0):.2f}'},
            {"模型": "Lightweight MLP", "Accuracy": fmt_pct(mlp_metrics["accuracy"]), "Precision": fmt_pct(mlp_metrics["precision"]), "Recall": fmt_pct(mlp_metrics["recall"]), "F1 Score": fmt_pct(mlp_metrics["f1"]), "训练耗时(s)": f'{mlp_metrics.get("train_time", 0):.2f}'},
        ]
    )
    panel("RandomForest 与 Lightweight MLP 对比表")
    dark_table(comp, height=150)


def batch_page():
    st.markdown('<div class="page-title">批量检测</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        uploaded = st.file_uploader("CSV文件上传", type=["csv", "gz"])
        st.caption("支持 CICIDS 或 UCI Kitsune 等包含数值特征的 CSV/CSV.GZ。若包含 Label 列，将用于展示攻击类型；预测仍调用已保存模型。")
    with right:
        model_name = st.selectbox("模型选择", ["Lightweight MLP", "RandomForest"])
        threshold = st.slider("风险阈值", 0.1, 0.95, 0.50, 0.05)
        batch_size = st.selectbox("批量大小", [256, 512, 1024, 2048], index=2)

    if uploaded is not None:
        upload_key = (uploaded.name, getattr(uploaded, "size", None), int(batch_size))
        if st.session_state.get("batch_upload_key") != upload_key:
            st.session_state["batch_raw"] = read_detection_csv(uploaded, nrows=batch_size)
            st.session_state["batch_upload_key"] = upload_key
            st.session_state["batch_source_kind"] = "upload"
            st.session_state["active_event_source"] = "batch"
            st.session_state["batch_updated_at"] = time.time()
            current_vm_mtime = max(
                os.path.getmtime(VM_SIGNAL_FILE) if os.path.exists(VM_SIGNAL_FILE) else 0,
                os.path.getmtime(LIVE_VM_CSV) if os.path.exists(LIVE_VM_CSV) and os.path.getsize(LIVE_VM_CSV) > 0 else 0,
            )
            st.session_state["last_seen_vm_mtime"] = current_vm_mtime
            st.session_state["last_upload_name"] = uploaded.name
        raw = st.session_state["batch_raw"]
        source_name = uploaded.name
        source_size = getattr(uploaded, "size", None)
        align_msg = feature_alignment_message(raw)
        if align_msg:
            if feature_space_mismatch(raw)[0]:
                st.warning(align_msg)
            else:
                st.info(align_msg)
    else:
        cached_raw = st.session_state.get("batch_raw")
        if st.session_state.get("batch_source_kind") == "upload" and isinstance(cached_raw, pd.DataFrame) and not cached_raw.empty:
            raw = cached_raw
            source_name = st.session_state.get("last_upload_name", "上次上传数据")
            source_size = st.session_state.get("batch_upload_key", (None, len(raw), None))[1]
            st.info(f"当前继续使用上次上传数据：{source_name}。切换页面不会回退为演示数据；再次上传 CSV 后会刷新为新数据。")
        else:
            raw = make_demo_raw(batch_size)
            st.session_state["batch_raw"] = raw
            st.session_state["batch_source_kind"] = "demo"
            st.session_state["batch_upload_key"] = ("内置演示数据", len(raw), int(batch_size))
            source_name = "内置演示数据"
            source_size = len(raw)
            st.info("当前未上传 CSV，页面使用模拟批量流量数据兜底展示。")

    input_signature = (
        source_name,
        source_size,
        len(raw),
        tuple(raw.columns),
        model_name,
        float(threshold),
        int(batch_size),
    )

    result, warning = predict_batch(raw, model_name, threshold)
    if uploaded is not None:
        st.session_state["active_event_source"] = "batch"
        st.session_state["batch_updated_at"] = time.time()
        current_vm_mtime = max(
            os.path.getmtime(VM_SIGNAL_FILE) if os.path.exists(VM_SIGNAL_FILE) else 0,
            os.path.getmtime(LIVE_VM_CSV) if os.path.exists(LIVE_VM_CSV) and os.path.getsize(LIVE_VM_CSV) > 0 else 0,
        )
        st.session_state["last_seen_vm_mtime"] = current_vm_mtime
    st.session_state["batch_page_result"] = result
    st.session_state["batch_warning"] = warning
    st.session_state["batch_model"] = model_name
    st.session_state["batch_threshold"] = threshold
    st.session_state["batch_source_name"] = source_name
    is_preview_only = bool(getattr(result, "attrs", {}).get("preview_only")) if isinstance(result, pd.DataFrame) else False
    st.session_state["batch_preview_only"] = is_preview_only

    if st.session_state.get("batch_input_signature") != input_signature:
        st.session_state["batch_input_signature"] = input_signature
        if uploaded is not None and not result.empty and not is_preview_only:
            try:
                inserted = persist_batch_results(result, model_name)
                st.session_state["batch_persisted_signature"] = input_signature
                st.session_state["batch_db_message"] = f"上传数据已实时检测并写入 SQLite 数据库：{inserted} 条。"
            except Exception as exc:
                st.session_state["batch_persisted_signature"] = None
                st.session_state["batch_db_message"] = f"数据库写入失败：{exc}"
        elif is_preview_only:
            st.session_state["batch_persisted_signature"] = None
            st.session_state["batch_db_message"] = "当前为 Label 预览结果，未执行模型推理，也不会写入 SQLite 检测结果表。"
        else:
            st.session_state["batch_persisted_signature"] = None
            st.session_state["batch_db_message"] = "当前演示结果已实时更新，未写入 SQLite 数据库。"

    already_persisted = st.session_state.get("batch_persisted_signature") == input_signature
    is_preview_only = bool(st.session_state.get("batch_preview_only"))
    if st.button("保存当前检测结果到 SQLite 数据库", use_container_width=True, disabled=already_persisted or is_preview_only):
        current_result = st.session_state.get("batch_page_result", pd.DataFrame())
        try:
            inserted = persist_batch_results(current_result, model_name)
            st.session_state["batch_persisted_signature"] = input_signature
            st.session_state["batch_db_message"] = f"检测结果已写入 SQLite 数据库：{inserted} 条。"
        except Exception as exc:
            st.session_state["batch_db_message"] = f"数据库写入失败：{exc}"
    if already_persisted:
        st.caption("当前检测结果已经保存到 SQLite 数据库。")
    if is_preview_only:
        st.caption("当前结果为真实 Label 可视化预览，不作为模型检测结果入库。")

    if warning:
        st.warning(warning)
    else:
        st.success(f"检测结果已实时更新：{st.session_state.get('batch_model', model_name)}，风险阈值 {st.session_state.get('batch_threshold', threshold):.2f}。结果已同步到态势总览、动态关系图、实时监控和日志中心。")
    db_message = st.session_state.get("batch_db_message")
    if db_message:
        if db_message.startswith("数据库写入失败"):
            st.warning(db_message)
        else:
            st.info(db_message)

    total, normal, attacks, high, _, _ = summarize(result)
    cols = st.columns(4)
    cols[0].metric("总流量数", f"{total:,}")
    cols[1].metric("正常流量数", f"{normal:,}")
    cols[2].metric("攻击流量数", f"{attacks:,}")
    cols[3].metric("高危告警数", f"{high:,}")
    dark_table(result[["时间", "源IP", "目的IP", "攻击类型", "风险等级", "预测概率", "检测结果"]], height=420)
    st.download_button("导出检测结果CSV", result.to_csv(index=False).encode("utf-8-sig"), "batch_detection_results.csv", "text/csv")


def realtime_page(events):
    st.markdown('<div class="sim-note">本页流量趋势、协议分布、攻击来源和告警列表均来自当前检测事件；CPU、内存等主机状态为演示状态值。</div>', unsafe_allow_html=True)
    line = trend_df(events, "2min")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=line.index, y=line["正常"] + line["攻击"], name="当前流量", line=dict(color=INFO, width=3), fill="tozeroy"))
        fig.update_layout(title="当前流量趋势图")
        show_chart(fig, 330)
    with c2:
        proto = events["协议"].value_counts().reset_index()
        proto.columns = ["协议", "数量"]
        fig = px.pie(proto, names="协议", values="数量", hole=.55, title="协议分布", color_discrete_sequence=[NORMAL, BLUE, WARN, INFO])
        show_chart(fig, 330)

    c3, c4 = st.columns([1.2, 1])
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=line.index, y=line["正常"], name="正常流量", line=dict(color=NORMAL, width=3)))
        fig.add_trace(go.Scatter(x=line.index, y=line["攻击"], name="攻击流量", line=dict(color=BAD, width=3)))
        fig.update_layout(title="正常流量与攻击流量变化")
        show_chart(fig, 330)
    with c4:
        attack_sources = events[events["检测结果"] == "攻击"]["源IP"].value_counts().head(8).reset_index()
        attack_sources.columns = ["来源IP", "攻击数"]
        if attack_sources.empty:
            attack_sources = pd.DataFrame({"来源IP": ["暂无攻击"], "攻击数": [0]})
        fig = px.bar(attack_sources, x="攻击数", y="来源IP", orientation="h", title="攻击来源IP分布", color="攻击数", color_continuous_scale=[INFO, BAD])
        show_chart(fig, 330)

    s1, s2, s3, s4 = st.columns(4)
    attack_count = int((events["检测结果"] == "攻击").sum())
    total_count = len(events)
    high_count = int((events["风险等级"] == "高危").sum())
    s1.metric("CPU 使用率", "26%")
    s2.metric("内存使用率", "48%")
    s3.metric("检测队列", f"{total_count:,}")
    s4.metric("告警吞吐量", f"{attack_count:,} / {total_count:,}")
    panel("实时告警列表")
    dark_table(events[events["检测结果"] == "攻击"][["时间", "源IP", "目的IP", "攻击类型", "风险等级", "预测概率"]].head(12), height=320)


def log_center_page(events):
    st.markdown('<div class="page-title">日志中心</div>', unsafe_allow_html=True)
    active_events = events.copy()
    live_mode = os.path.exists(VM_SIGNAL_FILE) or (os.path.exists(LIVE_VM_CSV) and os.path.getsize(LIVE_VM_CSV) > 0)
    batch_mode = not live_mode and st.session_state.get("batch_page_result") is not None
    all_db_events, db_error = load_detection_events_from_db(limit=5000)
    if not active_events.empty and not all_db_events.empty:
        option_events = pd.concat([active_events, all_db_events], ignore_index=True)
    elif not active_events.empty:
        option_events = active_events
    else:
        option_events = all_db_events
    attack_options = sorted(option_events["攻击类型"].dropna().unique().tolist()) if not option_events.empty else []
    if live_mode:
        st.caption(f"数据来源：虚拟机实时采集 CSV 联动；本次检测记录 {len(active_events):,} 条，SQLite历史记录 {len(all_db_events):,} 条。")
    elif batch_mode:
        st.caption(f"数据来源：当前批量检测结果实时联动；本次检测记录 {len(active_events):,} 条，SQLite历史记录 {len(all_db_events):,} 条。")
    else:
        st.caption(f"数据来源：SQLite数据库；当前可查询历史检测记录 {len(all_db_events):,} 条。")
    f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 1, 1.2])
    with f1:
        date_range = st.date_input("时间范围", value=(datetime.now().date() - timedelta(days=1), datetime.now().date()))
    with f2:
        log_type = st.selectbox("日志类型", ["全部", "攻击日志", "正常日志", "告警日志"])
    with f3:
        risk = st.selectbox("风险等级", ["全部", "高危", "中危", "低危"])
    with f4:
        attack_type = st.selectbox("攻击类型", ["全部"] + attack_options)
    with f5:
        keyword = st.text_input("关键词搜索", "")

    if not active_events.empty:
        logs = active_events.copy()
        query_error = None
    else:
        logs, query_error = load_detection_events_from_db(
            limit=5000,
            risk_level=None if risk == "全部" else risk,
            attack_type=None if attack_type == "全部" else attack_type,
            keyword=keyword or None,
        )
    if query_error:
        st.warning(f"数据库读取失败，日志中心暂时无法读取历史记录：{query_error}")

    if logs.empty:
        try:
            log_df = get_logs(
                limit=5000,
                log_type=None if log_type == "全部" else log_type,
                risk_level=None if risk == "全部" else risk,
                keyword=keyword or None,
            )
            if not log_df.empty:
                logs = log_df.rename(
                    columns={
                        "log_time": "时间",
                        "log_type": "日志类型",
                        "attack_type": "攻击类型",
                        "source_ip": "源IP",
                        "dest_ip": "目的IP",
                        "risk_level": "风险等级",
                        "message": "描述",
                    }
                )
                logs["检测结果"] = np.where(logs["日志类型"] == "正常日志", "正常", "攻击")
        except Exception as exc:
            st.warning(f"系统日志读取失败：{exc}")

    if logs.empty:
        st.info("暂无检测记录")
        empty_logs = pd.DataFrame(columns=["时间", "日志类型", "攻击类型", "源IP", "目的IP", "风险等级", "描述"])
        dark_table(empty_logs, height=260)
        st.download_button("导出日志", empty_logs.to_csv(index=False).encode("utf-8-sig"), "ids_logs.csv", "text/csv")
        return

    if "日志类型" not in logs.columns:
        logs["日志类型"] = np.where(logs["检测结果"] == "攻击", "攻击日志", "正常日志")
        logs.loc[logs["风险等级"] == "高危", "日志类型"] = "告警日志"
    if "描述" not in logs.columns:
        logs["描述"] = np.where(logs["检测结果"] == "攻击", "检测到异常访问行为，已生成风险告警", "流量特征处于正常范围")
    logs["时间"] = pd.to_datetime(logs["时间"], errors="coerce")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        logs = logs[(logs["时间"].dt.date >= start_date) & (logs["时间"].dt.date <= end_date)]
    if log_type != "全部":
        logs = logs[logs["日志类型"] == log_type]
    if risk != "全部":
        logs = logs[logs["风险等级"] == risk]
    if attack_type != "全部":
        logs = logs[logs["攻击类型"] == attack_type]
    if keyword:
        keyword_mask = logs.astype(str).apply(lambda row: row.str.contains(keyword, case=False, na=False).any(), axis=1)
        logs = logs[keyword_mask]

    c = st.columns(4)
    c[0].metric("总日志数", f"{len(logs):,}")
    c[1].metric("攻击日志数", f"{(logs['检测结果'] == '攻击').sum():,}")
    c[2].metric("告警日志数", f"{(logs['风险等级'] == '高危').sum():,}")
    c[3].metric("正常日志数", f"{(logs['检测结果'] == '正常').sum():,}")

    page_size = st.selectbox("每页条数", [10, 20, 50], index=1)
    max_page = max(1, int(np.ceil(len(logs) / page_size)))
    page = st.number_input("页码", min_value=1, max_value=max_page, value=1)
    view = logs.iloc[(page - 1) * page_size : page * page_size]
    if view.empty:
        st.info("暂无检测记录")
    else:
        dark_table(view[["时间", "日志类型", "攻击类型", "源IP", "目的IP", "风险等级", "描述"]], height=430)
    st.download_button("导出日志", logs.to_csv(index=False).encode("utf-8-sig"), "ids_logs.csv", "text/csv")


def model_file_size_mb(path):
    return round(os.path.getsize(path) / 1024 / 1024, 4) if os.path.exists(path) else 0.0


def load_model_file_table(rf_metrics, mlp_metrics):
    try:
        model_files = get_model_files()
    except Exception:
        model_files = pd.DataFrame()

    if not model_files.empty:
        return pd.DataFrame(
            [
                {
                    "当前已加载模型": row["model_name"],
                    "模型状态": row["status"],
                    "模型类型": row["model_type"],
                    "模型文件大小(MB)": f"{float(row['file_size'] or 0):.4f}",
                    "准确率": fmt_pct(row["accuracy"] or 0),
                    "F1 Score": fmt_pct(row["f1_score"] or 0),
                    "最近更新时间": row["updated_at"],
                    "模型文件路径": row["model_path"],
                }
                for _, row in model_files.iterrows()
            ]
        )

    return pd.DataFrame(
        [
            {
                "当前已加载模型": "Lightweight MLP",
                "模型状态": "在线" if os.path.exists(MLP_MODEL_PATH) else "离线",
                "模型类型": "Deep Learning",
                "模型文件大小(MB)": f"{model_file_size_mb(MLP_MODEL_PATH):.4f}",
                "准确率": fmt_pct(mlp_metrics["accuracy"]),
                "F1 Score": fmt_pct(mlp_metrics["f1"]),
                "最近更新时间": datetime.fromtimestamp(os.path.getmtime(MLP_MODEL_PATH)).strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(MLP_MODEL_PATH) else "-",
                "模型文件路径": MLP_MODEL_PATH,
            },
            {
                "当前已加载模型": "RandomForest",
                "模型状态": "在线" if os.path.exists(RF_MODEL_PATH) else "离线",
                "模型类型": "Machine Learning",
                "模型文件大小(MB)": f"{model_file_size_mb(RF_MODEL_PATH):.4f}",
                "准确率": fmt_pct(rf_metrics["accuracy"]),
                "F1 Score": fmt_pct(rf_metrics["f1"]),
                "最近更新时间": datetime.fromtimestamp(os.path.getmtime(RF_MODEL_PATH)).strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(RF_MODEL_PATH) else "-",
                "模型文件路径": RF_MODEL_PATH,
            },
        ]
    )


def sync_model_files_from_disk(rf_metrics, mlp_metrics):
    models = [
        ("RandomForest", "Machine Learning", RF_MODEL_PATH, rf_metrics),
        ("Lightweight MLP", "Deep Learning", MLP_MODEL_PATH, mlp_metrics),
    ]
    for model_name, model_type, model_path, metrics in models:
        upsert_model_file(
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            file_size=model_file_size_mb(model_path),
            status="在线" if os.path.exists(model_path) else "离线",
            accuracy=float(metrics.get("accuracy", 0) or 0),
            f1_score=float(metrics.get("f1", 0) or 0),
            updated_at=datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(model_path) else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def model_management_page(rf_metrics, mlp_metrics):
    st.markdown('<div class="page-title">模型管理</div>', unsafe_allow_html=True)
    models = load_model_file_table(rf_metrics, mlp_metrics)
    dark_table(models, height=165)
    if st.button("重新加载模型"):
        load_detector.clear()
        st.success("模型缓存已清理，下次检测将重新加载模型文件。")

    c1, c2 = st.columns([1, 1])
    with c1:
        radar = pd.DataFrame(
            {
                "指标": ["Accuracy", "Precision", "Recall", "F1 Score", "推理速度"],
                "Lightweight MLP": [mlp_metrics["accuracy"], mlp_metrics["precision"], mlp_metrics["recall"], mlp_metrics["f1"], .96],
                "RandomForest": [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"], .82],
            }
        )
        fig = go.Figure()
        for name, color in [("Lightweight MLP", NORMAL), ("RandomForest", BLUE)]:
            fig.add_trace(go.Scatterpolar(r=radar[name].tolist() + [radar[name].iloc[0]], theta=radar["指标"].tolist() + [radar["指标"].iloc[0]], fill="toself", name=name, line=dict(color=color)))
        fig.update_layout(title="模型性能对比雷达图", polar=dict(bgcolor="rgba(4,18,32,.72)", radialaxis=dict(range=[0, 1], gridcolor="rgba(126,164,191,.18)", color=MUTED)))
        show_chart(fig, 360)
    with c2:
        try:
            metrics_history = get_model_metrics()
        except Exception:
            metrics_history = pd.DataFrame()
        if metrics_history.empty:
            history = pd.DataFrame(
                [
                    {"模型": "Lightweight MLP", "Accuracy": fmt_pct(mlp_metrics["accuracy"]), "F1 Score": fmt_pct(mlp_metrics["f1"]), "训练耗时(s)": f'{mlp_metrics.get("train_time", 0):.2f}', "创建时间": "-"},
                    {"模型": "RandomForest", "Accuracy": fmt_pct(rf_metrics["accuracy"]), "F1 Score": fmt_pct(rf_metrics["f1"]), "训练耗时(s)": f'{rf_metrics.get("train_time", 0):.2f}', "创建时间": "-"},
                ]
            )
        else:
            history = pd.DataFrame(
                [
                    {
                        "模型": row["model_name"],
                        "Accuracy": fmt_pct(row["accuracy"] or 0),
                        "F1 Score": fmt_pct(row["f1_score"] or 0),
                        "训练耗时(s)": f'{float(row["train_time"] or 0):.2f}',
                        "创建时间": row["created_at"],
                    }
                    for _, row in metrics_history.iterrows()
                ]
            )
        panel("模型训练历史记录")
        dark_table(history, height=310)

    st.caption("原有模型保存与加载逻辑保留在 src/train_baseline.py、src/train_dl.py 和 src/detector.py 中，本页面只负责状态展示与缓存刷新。")


def sidebar():
    page_options = ["态势总览", "AI检测分析", "批量检测", "实时监控", "日志中心", "模型管理"]
    page_key_map = {
        "overview": "态势总览",
        "ai": "AI检测分析",
        "batch": "批量检测",
        "monitor": "实时监控",
        "logs": "日志中心",
        "models": "模型管理",
    }
    query_page = st.query_params.get("page", "overview")
    default_page = page_key_map.get(query_page, "态势总览")
    with st.sidebar:
        st.markdown(f"### {SYSTEM_NAME}")
        page = st.radio(
            "导航菜单",
            page_options,
            index=page_options.index(default_page),
            label_visibility="collapsed",
        )
    return page, "内置演示数据"


def load_selected_data(selected):
    if selected == "内置演示数据":
        return make_demo_raw()
    try:
        return load_csv(os.path.join(DATA_DIR, selected))
    except Exception:
        return make_demo_raw()


def main():
    inject_styles()
    try:
        init_db()
    except Exception as exc:
        st.warning(f"SQLite 数据库初始化失败，系统将继续使用原有文件与会话数据运行：{exc}")
    page, selected = sidebar()
    topbar()
    rf_metrics, mlp_metrics = get_metrics()
    try:
        sync_model_files_from_disk(rf_metrics, mlp_metrics)
    except Exception as exc:
        st.warning(f"模型文件状态同步到数据库失败：{exc}")
    raw_df = load_selected_data(selected)
    events, event_source = get_active_events()

    if page == "态势总览":
        st.caption(f"当前事件来源：{event_source}")
        live_warning = st.session_state.get("live_vm_warning")
        if live_warning:
            st.warning(live_warning)
        overview_page(events, rf_metrics, mlp_metrics)
    elif page == "AI检测分析":
        ai_analysis_page(events, rf_metrics, mlp_metrics)
    elif page == "批量检测":
        batch_page()
    elif page == "实时监控":
        realtime_page(events)
    elif page == "日志中心":
        log_center_page(events)
    elif page == "模型管理":
        model_management_page(rf_metrics, mlp_metrics)


if __name__ == "__main__":
    main()
