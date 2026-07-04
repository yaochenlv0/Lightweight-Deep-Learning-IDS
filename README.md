# Lightweight Deep Learning Intrusion Detection System

This repository contains a Streamlit-based intrusion detection demo system for a
graduation project. It integrates a RandomForest baseline model, a lightweight
MLP model, batch CSV detection, risk scoring, alert visualization, log
management, model status management, and virtual-machine attack demo linkage.

## Recommended Repository Name

`Lightweight-Deep-Learning-IDS`

## Features

- Situation overview dashboard with traffic statistics and alert topology
- AI detection analysis with metrics, confusion matrices, loss curve, and feature importance
- Batch CSV detection and result export
- Real-time monitoring view for alert events and protocol distribution
- Log center with filtering and export
- Model management page for model file status and metrics
- VMware/Kali/Ubuntu demo linkage through scenario signal files

## Project Structure

```text
.
├── app.py                  # Streamlit application entry
├── main.py                 # Training/evaluation workflow entry
├── config.py               # Path and model configuration
├── requirements.txt        # Python dependencies
├── src/                    # Data processing, training, detection, database helpers
├── models/                 # Trained RF/MLP models and scaler
├── results/                # Metrics and visualization outputs
└── data/demo_scenarios/    # Small demo CSV files for VM scenario linkage
```

Large raw datasets are not included in the GitHub repository. Place CICIDS2017
or Kitsune raw files under `data/cicids/` or `data/kitsune/` if retraining is
required.

## Quick Start

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
python -m streamlit run app.py --server.port 8501
```

Open the browser:

```text
http://127.0.0.1:8501
```

## Virtual Machine Demo Signal

The VM demo mode reads a signal file from:

```text
data/vm_bridge/current_scenario.json
```

Example:

```json
{
  "scenario": "ssh_patator",
  "attack_ip": "192.168.99.141",
  "target_ip": "192.168.99.140",
  "rows": 200
}
```

Supported scenario names:

- `normal`
- `portscan`
- `ssh_patator`
- `ftp_patator`
- `dos`
- `mixed_attack`

## Model Files

The repository includes the trained model files needed for direct demo running:

- `models/rf_model.pkl`
- `models/mlp_model.pth`
- `models/scaler.pkl`

## Notes

- The SQLite database file is generated automatically at runtime.
- Raw datasets, local logs, virtual environments, thesis documents, and generated
  presentation/document outputs are intentionally ignored.
