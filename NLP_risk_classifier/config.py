from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/NLP_risk_classifier")
DATA_DIR = BASE_DIR / "data"
CONTROL_DATA_DIR = DATA_DIR / "controls"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
CONTROL_MODEL_DIR = MODEL_DIR / "controls"

# ---------------------------
# 6 Control Tasks
# ---------------------------
TASKS = [
    "Control_Fail_(2)",
    "Control_Failed",
    "Control_name",
    "What_Control_Failed",
    "Control_Time",
    "Control_Strength",
]

# ---------------------------
# Label Column Names
# ---------------------------
LABEL_COL_MAP = {
    "Control_Fail_(2)": "Control Fail (2)_label",
    "Control_Failed": "Control Failed_label",
    "Control_name": "Control name_label",
    "What_Control_Failed": "What Control Failed_label",
    "Control_Time": "Control Time_label",
    "Control_Strength": "Control Strength_label",
}
