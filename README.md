# 基于深度学习的轻量化入侵检测系统

本项目是基于 Streamlit 的轻量化入侵检测可视化系统，保留 RandomForest 基线模型、Lightweight MLP 深度学习模型、批量检测、指标展示、日志中心和模型管理等功能。

## 目录结构

- `app.py`：Streamlit 可视化系统入口。
- `main.py`：模型训练与评估流程入口。
- `config.py`：路径与核心配置。
- `src/`：数据预处理、模型训练、检测、评估与可视化辅助代码。
- `models/`：已训练模型文件与标准化器。
- `results/`：模型指标与可视化结果文件。
- `requirements.txt`：项目 Python 依赖。

## 运行方式

```powershell
cd "D:\作业\大四上\毕设\基于轻量化深度学习的入侵检测系统\基于深度学习的轻量化入侵检测系统"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

浏览器访问：

```text
http://localhost:8501/
```

## 环境说明

当前项目已配置独立虚拟环境 `.venv`，并已安装 `requirements.txt` 中依赖，包括 Streamlit、Plotly、PyTorch、scikit-learn、pandas、numpy 等。
