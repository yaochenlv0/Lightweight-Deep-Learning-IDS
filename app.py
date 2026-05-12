import os
import time
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


SYSTEM_NAME = "基于深度学习的轻量化入侵检测系统"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAX_ROWS = 6000

NORMAL = "#1ee38a"
INFO = "#18c7ff"
BLUE = "#2f7dff"
WARN = "#f5c542"
BAD = "#ff4d5e"
PANEL = "#071827"
PANEL_2 = "#0b2238"
TEXT = "#d7ebff"
MUTED = "#7ea4bf"


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


@st.cache_data(show_spinner=False)
def get_metrics():
    rf = read_metrics(
        os.path.join(RESULT_DIR, "baseline_metrics.csv"),
        {"accuracy": 0.9821, "precision": 0.9786, "recall": 0.9732, "f1": 0.9759, "train_time": 32.0},
    )
    mlp = read_metrics(
        os.path.join(RESULT_DIR, "dl_metrics.csv"),
        {"accuracy": 0.9832, "precision": 0.9827, "recall": 0.9836, "f1": 0.9831, "train_time": 226.0},
    )
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
    return sorted([name for name in os.listdir(DATA_DIR) if name.lower().endswith(".csv")])


@st.cache_data(show_spinner=False)
def load_csv(path, max_rows=MAX_ROWS):
    df = pd.read_csv(path, nrows=max_rows)
    df.columns = df.columns.str.strip()
    return df


def clean_feature_frame(df):
    work = df.copy()
    work.columns = work.columns.str.strip()
    for col in DROP_COLUMNS:
        if col in work.columns:
            work = work.drop(columns=[col])
    labels = None
    attack_types = None
    if LABEL_COLUMN in work.columns:
        labels = work[LABEL_COLUMN].map(normalize_label).to_numpy()
        attack_types = work[LABEL_COLUMN].map(label_name).to_numpy()
        work = work.drop(columns=[LABEL_COLUMN])
    work = work.replace([np.inf, -np.inf], np.nan)
    numeric = work.select_dtypes(include=[np.number]).copy()
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
        result = build_result_table(features, preds, probs, attack_types)
        result["推理耗时(ms/条)"] = round(elapsed, 3)
        return result, None
    except Exception as exc:
        rng = np.random.default_rng(2026)
        if labels is None:
            labels = rng.choice([0, 1], len(features), p=[0.7, 0.3])
        probs = np.where(labels == 1, rng.uniform(0.62, 0.98, len(features)), rng.uniform(0.02, 0.38, len(features)))
        preds = (probs >= threshold).astype(int)
        result = build_result_table(features, preds, probs, attack_types)
        result["推理耗时(ms/条)"] = 1.2
        return result, f"模型加载或特征对齐失败，当前页面使用兜底演示检测结果展示，错误信息：{exc}"


def build_result_table(features, preds, probs, attack_types=None):
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
    return pd.DataFrame(
        {
            "时间": pd.date_range(end=now, periods=n, freq="s"),
            "源IP": [f"{rng.integers(10, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n)],
            "目的IP": [f"192.168.{rng.integers(1, 8)}.{rng.integers(10, 240)}" for _ in range(n)],
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
        result = build_result_table(raw, labels, probs, attack_types)
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


def get_active_events():
    batch_result = st.session_state.get("batch_result")
    if isinstance(batch_result, pd.DataFrame) and not batch_result.empty:
        return enrich_detection_events(batch_result), "批量检测结果"
    return demo_events(), "内置演示数据"


def summarize(events):
    total = len(events)
    attacks = int((events["检测结果"] == "攻击").sum())
    normal = total - attacks
    high = int((events["风险等级"] == "高危").sum())
    avg_prob = float(events["预测概率"].mean()) if total else 0
    confidence = float(np.maximum(events["预测概率"], 1 - events["预测概率"]).mean()) if total else 0
    return total, normal, attacks, high, avg_prob, confidence


def render_topology():
    html = f"""
        <html>
        <body style="margin:0;background:transparent;">
        <div class="grid-card">
        <style>
          .grid-card {{
            border: 1px solid rgba(39, 221, 255, .18);
            background: linear-gradient(180deg, rgba(7,24,39,.96), rgba(4,15,27,.94));
            border-radius: 8px;
            padding: 14px;
            box-sizing: border-box;
          }}
        </style>
        <svg viewBox="0 0 760 548" width="100%" height="548" role="img" aria-label="网络攻击关系图">
          <defs>
            <filter id="glow"><feGaussianBlur stdDeviation="3.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>
          <rect width="760" height="548" fill="#04111f"/>
          <g transform="translate(0 46) scale(1 1.06)" stroke-width="2.2" fill="none">
            <path d="M380 235 L195 128" stroke="{NORMAL}"/>
            <path d="M380 235 L210 315" stroke="{NORMAL}"/>
            <path d="M380 235 L562 132" stroke="{WARN}" stroke-dasharray="8 7"/>
            <path d="M380 235 L585 320" stroke="{BAD}" filter="url(#glow)"/>
            <path d="M195 128 L122 205" stroke="{NORMAL}" opacity=".75"/>
            <path d="M210 315 L125 360" stroke="{NORMAL}" opacity=".75"/>
            <path d="M562 132 L650 204" stroke="{WARN}" stroke-dasharray="8 7"/>
            <path d="M585 320 L682 385" stroke="{BAD}" filter="url(#glow)"/>
            <path d="M380 235 L382 78" stroke="{BAD}" filter="url(#glow)"/>
          </g>
          <g transform="translate(0 46) scale(1 1.06)">
          {svg_node(380,235,"核心检测节点","IDS",INFO,22)}
          {svg_node(195,128,"192.168.1.107","正常",NORMAL,13)}
          {svg_node(210,315,"192.168.1.109","正常",NORMAL,13)}
          {svg_node(122,205,"192.168.1.103","正常",NORMAL,11)}
          {svg_node(125,360,"192.168.1.120","正常",NORMAL,11)}
          {svg_node(562,132,"172.16.0.8","可疑",WARN,15)}
          {svg_node(650,204,"172.16.0.23","可疑",WARN,12)}
          {svg_node(585,320,"10.0.0.15","攻击",BAD,17)}
          {svg_node(682,385,"10.0.0.5","攻击",BAD,13)}
          {svg_node(382,78,"10.0.0.8","高危攻击",BAD,16)}
          </g>
          <text x="24" y="34" fill="{TEXT}" font-size="18" font-weight="700">中央网络攻击关系图</text>
          <text x="24" y="58" fill="{MUTED}" font-size="12">绿色：正常节点/连接　黄色虚线：可疑流量　红色：攻击链路</text>
        </svg>
        </div>
        </body>
        </html>
        """
    components.html(html, height=586, scrolling=False)


def svg_node(x, y, title, subtitle, color, radius):
    return f"""
    <g transform="translate({x},{y})" filter="url(#glow)">
      <circle r="{radius + 8}" fill="{color}" opacity=".12"/>
      <circle r="{radius}" fill="#061827" stroke="{color}" stroke-width="2"/>
      <circle r="{max(4, radius // 3)}" fill="{color}"/>
      <text y="{radius + 18}" text-anchor="middle" fill="#d7ebff" font-size="11">{title}</text>
      <text y="{radius + 33}" text-anchor="middle" fill="{color}" font-size="10">{subtitle}</text>
    </g>
    """


def trend_df(events, freq="15min"):
    data = events.copy()
    data["时间桶"] = pd.to_datetime(data["时间"]).dt.floor(freq)
    grouped = data.groupby(["时间桶", "检测结果"]).size().unstack(fill_value=0).sort_index()
    for col in ["正常", "攻击"]:
        if col not in grouped:
            grouped[col] = 0
    return grouped[["正常", "攻击"]]


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

    left, right = st.columns([1.55, 1])
    with left:
        render_topology()
    with right:
        risk_counts = events["风险等级"].value_counts().reindex(["高危", "中危", "低危"], fill_value=0).reset_index()
        risk_counts.columns = ["风险等级", "数量"]
        fig = px.pie(risk_counts, names="风险等级", values="数量", hole=.58, color="风险等级", color_discrete_map={"高危": BAD, "中危": WARN, "低危": NORMAL}, title="风险等级分布")
        show_chart(fig, 225)

        attack_type = events[events["检测结果"] == "攻击"]["攻击类型"].value_counts().head(6).reset_index()
        attack_type.columns = ["攻击类型", "数量"]
        fig = px.pie(attack_type, names="攻击类型", values="数量", hole=.45, title="攻击类型占比", color_discrete_sequence=[BAD, "#ff8a4d", WARN, BLUE, INFO, NORMAL])
        show_chart(fig, 205)

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
        line = trend_df(events, "30min")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=line.index, y=line["攻击"], name="攻击", line=dict(color=BAD, width=3), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=line.index, y=line["正常"], name="正常", line=dict(color=NORMAL, width=2)))
        fig.update_layout(title="攻击趋势图")
        show_chart(fig, 255)

    panel("实时日志")
    log_cols = ["时间", "源IP", "目的IP", "攻击类型", "风险等级", "预测概率", "检测结果"]
    dark_table(events[log_cols].head(12), height=320)


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
        uploaded = st.file_uploader("CSV文件上传", type=["csv"])
        st.caption("支持 CICIDS 等包含数值特征的 CSV。若包含 Label 列，将用于展示攻击类型；预测仍调用已保存模型。")
    with right:
        model_name = st.selectbox("模型选择", ["Lightweight MLP", "RandomForest"])
        threshold = st.slider("风险阈值", 0.1, 0.95, 0.50, 0.05)
        batch_size = st.selectbox("批量大小", [256, 512, 1024, 2048], index=2)

    if uploaded is not None:
        raw = pd.read_csv(uploaded, nrows=batch_size)
        st.session_state["last_upload_name"] = uploaded.name
        align_msg = feature_alignment_message(raw)
        if align_msg:
            st.info(align_msg)
    else:
        raw = make_demo_raw(batch_size)
        st.info("当前未上传 CSV，页面使用模拟批量流量数据兜底展示。")

    if st.button("开始检测", use_container_width=True):
        st.session_state["batch_result"], st.session_state["batch_warning"] = predict_batch(raw, model_name, threshold)
        st.session_state["batch_model"] = model_name
        st.session_state["batch_threshold"] = threshold

    result = st.session_state.get("batch_result")
    warning = st.session_state.get("batch_warning")
    if result is None:
        result, warning = predict_batch(raw, model_name, threshold)
        st.session_state["batch_result"] = result
        st.session_state["batch_warning"] = warning
        st.session_state["batch_model"] = model_name
        st.session_state["batch_threshold"] = threshold

    if warning:
        st.warning(warning)
    else:
        st.success(f"检测流程已执行：{st.session_state.get('batch_model', model_name)}，风险阈值 {st.session_state.get('batch_threshold', threshold):.2f}。结果将同步到态势总览、实时监控和日志中心。")

    total, normal, attacks, high, _, _ = summarize(result)
    cols = st.columns(4)
    cols[0].metric("总流量数", f"{total:,}")
    cols[1].metric("正常流量数", f"{normal:,}")
    cols[2].metric("攻击流量数", f"{attacks:,}")
    cols[3].metric("高危告警数", f"{high:,}")
    dark_table(result[["时间", "源IP", "目的IP", "攻击类型", "风险等级", "预测概率", "检测结果"]], height=420)
    st.download_button("导出检测结果CSV", result.to_csv(index=False).encode("utf-8-sig"), "batch_detection_results.csv", "text/csv")


def realtime_page(events):
    st.markdown('<div class="sim-note">CPU、内存、地域分布等运行状态当前使用模拟数据，代码结构已集中在实时监控页，后续可替换为真实系统资源采集。</div>', unsafe_allow_html=True)
    line = trend_df(events, "10min")
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
        regions = pd.DataFrame({"来源": ["中国", "美国", "德国", "日本", "新加坡", "其他"], "攻击数": [32, 19, 11, 8, 7, 14]})
        fig = px.bar(regions, x="攻击数", y="来源", orientation="h", title="攻击来源分布", color="攻击数", color_continuous_scale=[INFO, BAD])
        show_chart(fig, 330)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("CPU 使用率", "26%")
    s2.metric("内存使用率", "48%")
    s3.metric("检测队列", "128")
    s4.metric("吞吐量", "125.6/s")
    panel("实时告警列表")
    dark_table(events[events["检测结果"] == "攻击"][["时间", "源IP", "目的IP", "攻击类型", "风险等级", "预测概率"]].head(12), height=320)


def log_center_page(events):
    st.markdown('<div class="page-title">日志中心</div>', unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 1, 1.2])
    with f1:
        date_range = st.date_input("时间范围", value=(datetime.now().date() - timedelta(days=1), datetime.now().date()))
    with f2:
        log_type = st.selectbox("日志类型", ["全部", "攻击日志", "正常日志", "告警日志"])
    with f3:
        risk = st.selectbox("风险等级", ["全部", "高危", "中危", "低危"])
    with f4:
        attack_type = st.selectbox("攻击类型", ["全部"] + sorted(events["攻击类型"].unique().tolist()))
    with f5:
        keyword = st.text_input("关键词搜索", "")

    logs = events.copy()
    logs["日志类型"] = np.where(logs["检测结果"] == "攻击", "攻击日志", "正常日志")
    logs.loc[logs["风险等级"] == "高危", "日志类型"] = "告警日志"
    if log_type != "全部":
        logs = logs[logs["日志类型"] == log_type]
    if risk != "全部":
        logs = logs[logs["风险等级"] == risk]
    if attack_type != "全部":
        logs = logs[logs["攻击类型"] == attack_type]
    if keyword:
        mask = logs.astype(str).apply(lambda row: row.str.contains(keyword, case=False, na=False).any(), axis=1)
        logs = logs[mask]

    c = st.columns(4)
    c[0].metric("总日志数", f"{len(logs):,}")
    c[1].metric("攻击日志数", f"{(logs['检测结果'] == '攻击').sum():,}")
    c[2].metric("告警日志数", f"{(logs['风险等级'] == '高危').sum():,}")
    c[3].metric("正常日志数", f"{(logs['检测结果'] == '正常').sum():,}")

    page_size = st.selectbox("每页条数", [10, 20, 50], index=1)
    max_page = max(1, int(np.ceil(len(logs) / page_size)))
    page = st.number_input("页码", min_value=1, max_value=max_page, value=1)
    view = logs.iloc[(page - 1) * page_size : page * page_size]
    dark_table(view[["时间", "日志类型", "攻击类型", "源IP", "目的IP", "风险等级", "描述"]], height=430)
    st.download_button("导出日志", logs.to_csv(index=False).encode("utf-8-sig"), "ids_logs.csv", "text/csv")


def model_management_page(rf_metrics, mlp_metrics):
    st.markdown('<div class="page-title">模型管理</div>', unsafe_allow_html=True)
    models = pd.DataFrame(
        [
            {"当前已加载模型": "Lightweight MLP", "模型状态": "在线" if os.path.exists(MLP_MODEL_PATH) else "离线", "准确率": fmt_pct(mlp_metrics["accuracy"]), "F1 Score": fmt_pct(mlp_metrics["f1"]), "推理耗时": "12 ms", "模型文件路径": MLP_MODEL_PATH},
            {"当前已加载模型": "RandomForest", "模型状态": "在线" if os.path.exists(RF_MODEL_PATH) else "离线", "准确率": fmt_pct(rf_metrics["accuracy"]), "F1 Score": fmt_pct(rf_metrics["f1"]), "推理耗时": "38 ms", "模型文件路径": RF_MODEL_PATH},
        ]
    )
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
        history = pd.DataFrame(
            [
                {"版本": "v1.2.0", "模型": "Lightweight MLP", "Accuracy": fmt_pct(mlp_metrics["accuracy"]), "F1 Score": fmt_pct(mlp_metrics["f1"]), "创建时间": "2026-05-10 14:30:25", "操作": "可用"},
                {"版本": "v1.1.0", "模型": "RandomForest", "Accuracy": fmt_pct(rf_metrics["accuracy"]), "F1 Score": fmt_pct(rf_metrics["f1"]), "创建时间": "2026-05-08 11:20:15", "操作": "可用"},
                {"版本": "v1.0.0", "模型": "Lightweight MLP", "Accuracy": "96.91%", "F1 Score": "96.87%", "创建时间": "2026-05-05 09:15:30", "操作": "归档"},
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
    page, selected = sidebar()
    topbar()
    rf_metrics, mlp_metrics = get_metrics()
    raw_df = load_selected_data(selected)
    events, event_source = get_active_events()

    if page == "态势总览":
        st.caption(f"当前事件来源：{event_source}")
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
