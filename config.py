import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "Tuesday-WorkingHours.pcap_ISCX.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_COLUMN = "Label"
DROP_COLUMNS = []

MLP_HIDDEN_LAYERS = [16, 8]
MLP_EPOCHS = 30
MLP_LR = 0.001
BATCH_SIZE = 64