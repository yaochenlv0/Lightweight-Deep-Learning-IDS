import os
import pandas as pd


def ensure_dir(path):
    """如果目录不存在，则自动创建"""
    os.makedirs(path, exist_ok=True)


def save_dict_to_csv(data_dict, save_path):
    """将字典保存为单行 CSV 文件"""
    df = pd.DataFrame([data_dict])
    df.to_csv(save_path, index=False, encoding="utf-8-sig")