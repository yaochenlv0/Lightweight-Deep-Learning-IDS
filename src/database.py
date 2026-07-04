import os
import sqlite3
from datetime import datetime

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "ids_system.db")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_value(row, names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _file_size_mb(path):
    if path and os.path.exists(path):
        return round(os.path.getsize(path) / 1024 / 1024, 4)
    return 0.0


def init_db():
    """Initialize SQLite tables used by the Streamlit result persistence layer."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                train_time REAL,
                model_file_size REAL,
                model_path TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT,
                model_name TEXT,
                source_ip TEXT,
                dest_ip TEXT,
                attack_type TEXT,
                risk_level TEXT,
                risk_score REAL,
                attack_probability REAL,
                detection_result TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_time TEXT,
                log_type TEXT,
                risk_level TEXT,
                attack_type TEXT,
                source_ip TEXT,
                dest_ip TEXT,
                message TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS model_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE,
                model_type TEXT,
                model_path TEXT,
                file_size REAL,
                status TEXT,
                accuracy REAL,
                f1_score REAL,
                updated_at TEXT
            );
            """
        )


def insert_model_metrics(metrics: dict):
    """Persist one model metric snapshot and update the model file status table."""
    init_db()
    created_at = metrics.get("created_at") or _now()
    model_name = metrics.get("model_name") or metrics.get("模型名称") or "Unknown"
    model_path = metrics.get("model_path") or metrics.get("模型路径")
    file_size = _safe_float(metrics.get("model_file_size"), _file_size_mb(model_path))
    accuracy = _safe_float(metrics.get("accuracy"))
    precision = _safe_float(metrics.get("precision_score", metrics.get("precision")))
    recall = _safe_float(metrics.get("recall"))
    f1_score = _safe_float(metrics.get("f1_score", metrics.get("f1")))
    train_time = _safe_float(metrics.get("train_time"))

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO model_metrics (
                model_name, accuracy, precision_score, recall, f1_score,
                train_time, model_file_size, model_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (model_name, accuracy, precision, recall, f1_score, train_time, file_size, model_path, created_at),
        )
        upsert_model_file(
            model_name=model_name,
            model_type=metrics.get("model_type") or ("Deep Learning" if "MLP" in model_name else "Machine Learning"),
            model_path=model_path,
            file_size=file_size,
            status=metrics.get("status", "在线" if model_path and os.path.exists(model_path) else "离线"),
            accuracy=accuracy,
            f1_score=f1_score,
            updated_at=created_at,
            conn=conn,
        )


def insert_detection_results(results_df):
    """Write batch detection results to detection_results and system_logs."""
    init_db()
    if results_df is None or results_df.empty:
        return 0

    created_at = _now()
    records = []
    log_records = []
    for _, row in results_df.iterrows():
        event_time = str(_first_value(row, ["event_time", "时间"], created_at))
        model_name = str(_first_value(row, ["model_name", "模型名称", "模型"], "Unknown"))
        detection_result = str(_first_value(row, ["detection_result", "检测结果"], "未知"))
        attack_type = str(_first_value(row, ["attack_type", "攻击类型"], "正常流量" if detection_result == "正常" else "未知攻击"))
        risk_level = str(_first_value(row, ["risk_level", "风险等级"], "低危"))
        source_ip = str(_first_value(row, ["source_ip", "源IP"], "0.0.0.0"))
        dest_ip = str(_first_value(row, ["dest_ip", "目的IP"], "0.0.0.0"))
        attack_probability = _safe_float(_first_value(row, ["attack_probability", "预测概率"], 0.0), 0.0)
        risk_score = _safe_float(_first_value(row, ["risk_score", "风险分"], attack_probability * 100), 0.0)

        records.append(
            (
                event_time,
                model_name,
                source_ip,
                dest_ip,
                attack_type,
                risk_level,
                risk_score,
                attack_probability,
                detection_result,
                created_at,
            )
        )

        log_type = "告警日志" if risk_level == "高危" else ("攻击日志" if detection_result == "攻击" else "正常日志")
        message = "检测到异常访问行为，已生成风险告警" if detection_result == "攻击" else "流量特征处于正常范围"
        log_records.append((event_time, log_type, risk_level, attack_type, source_ip, dest_ip, message, created_at))

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO detection_results (
                event_time, model_name, source_ip, dest_ip, attack_type,
                risk_level, risk_score, attack_probability, detection_result, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.executemany(
            """
            INSERT INTO system_logs (
                log_time, log_type, risk_level, attack_type,
                source_ip, dest_ip, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            log_records,
        )
    return len(records)


def insert_log(log_type, risk_level, message, attack_type=None, source_ip=None, dest_ip=None):
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO system_logs (
                log_time, log_type, risk_level, attack_type,
                source_ip, dest_ip, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now, log_type, risk_level, attack_type, source_ip, dest_ip, message, now),
        )


def _query_df(sql, params=()):
    init_db()
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_detection_results(limit=100, risk_level=None, attack_type=None, keyword=None):
    conditions = []
    params = []
    if risk_level and risk_level != "全部":
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if attack_type and attack_type != "全部":
        conditions.append("attack_type = ?")
        params.append(attack_type)
    if keyword:
        like = f"%{keyword}%"
        conditions.append(
            "(source_ip LIKE ? OR dest_ip LIKE ? OR attack_type LIKE ? OR risk_level LIKE ? OR detection_result LIKE ? OR model_name LIKE ?)"
        )
        params.extend([like] * 6)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(int(limit))
    return _query_df(f"SELECT * FROM detection_results {where} ORDER BY event_time DESC, id DESC LIMIT ?", params)


def get_logs(limit=100, log_type=None, risk_level=None, keyword=None):
    conditions = []
    params = []
    if log_type and log_type != "全部":
        conditions.append("log_type = ?")
        params.append(log_type)
    if risk_level and risk_level != "全部":
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if keyword:
        like = f"%{keyword}%"
        conditions.append("(message LIKE ? OR source_ip LIKE ? OR dest_ip LIKE ? OR attack_type LIKE ? OR log_type LIKE ?)")
        params.extend([like] * 5)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(int(limit))
    return _query_df(f"SELECT * FROM system_logs {where} ORDER BY log_time DESC, id DESC LIMIT ?", params)


def get_model_metrics():
    return _query_df(
        """
        SELECT mm.*
        FROM model_metrics mm
        INNER JOIN (
            SELECT model_name, MAX(id) AS max_id
            FROM model_metrics
            GROUP BY model_name
        ) latest ON latest.max_id = mm.id
        ORDER BY mm.created_at DESC, mm.id DESC
        """
    )


def upsert_model_file(
    model_name,
    model_type,
    model_path,
    file_size=None,
    status=None,
    accuracy=None,
    f1_score=None,
    updated_at=None,
    conn=None,
):
    if conn is None:
        init_db()
    file_size = _safe_float(file_size, _file_size_mb(model_path))
    status = status or ("在线" if model_path and os.path.exists(model_path) else "离线")
    updated_at = updated_at or _now()
    params = (model_name, model_type, model_path, file_size, status, accuracy, f1_score, updated_at)
    sql = """
        INSERT INTO model_files (
            model_name, model_type, model_path, file_size, status, accuracy, f1_score, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET
            model_type=excluded.model_type,
            model_path=excluded.model_path,
            file_size=excluded.file_size,
            status=excluded.status,
            accuracy=excluded.accuracy,
            f1_score=excluded.f1_score,
            updated_at=excluded.updated_at
    """
    if conn is not None:
        conn.execute(sql, params)
        return
    with _connect() as local_conn:
        local_conn.execute(sql, params)


def get_model_files():
    return _query_df("SELECT * FROM model_files ORDER BY updated_at DESC, id DESC")


def clear_demo_data():
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM detection_results")
        conn.execute("DELETE FROM system_logs")
