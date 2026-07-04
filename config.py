import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(BASE_DIR, "data", "Tuesday-WorkingHours.pcap_ISCX.csv")
DATASET_TYPE = os.getenv("IDS_DATASET_TYPE", "auto").lower()
KITSUNE_DATA_DIR = os.getenv("KITSUNE_DATA_DIR", os.path.join(DATA_DIR, "kitsune"))
KITSUNE_ATTACK = os.getenv("KITSUNE_ATTACK", "")
KITSUNE_MAX_ROWS = int(os.getenv("KITSUNE_MAX_ROWS", "200000"))
KITSUNE_SAMPLE_MODE = os.getenv("KITSUNE_SAMPLE_MODE", "natural").lower()
KITSUNE_5TUPLE_DIR = os.getenv("KITSUNE_5TUPLE_DIR", os.path.join(DATA_DIR, "kitsune"))
KITSUNE_5TUPLE_FILE = os.getenv("KITSUNE_5TUPLE_FILE", "")
CICIDS_DATA_DIR = os.getenv("CICIDS_DATA_DIR", os.path.join(DATA_DIR, "cicids"))
CICIDS_FILES = os.getenv("CICIDS_FILES", "")
CICIDS_MAX_ROWS = int(os.getenv("CICIDS_MAX_ROWS", "200000"))
CICIDS_SAMPLE_MODE = os.getenv("CICIDS_SAMPLE_MODE", "natural").lower()
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_COLUMN = "Label"
DROP_COLUMNS = []

MLP_HIDDEN_LAYERS = [32, 16]
MLP_EPOCHS = 30
MLP_LR = 0.001
BATCH_SIZE = 64
