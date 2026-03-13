import os

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.getcwd()


# =========================================================
# DATA PATHS
# =========================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "construction_safety_dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, "labels")

VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
VAL_LABELS_DIR = os.path.join(VAL_DIR, "labels")


# =========================================================
# CONFIG PATHS
# =========================================================

CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

MAIN_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
DATASET_CONFIG_FILE = os.path.join(CONFIG_DIR, "dataset.yaml")


# =========================================================
# MODEL PATHS
# =========================================================

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_model.pt")


# =========================================================
# LOGGING
# =========================================================

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE_NAME = "running_logs.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)


# =========================================================
# IMAGE SETTINGS
# =========================================================

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

DEFAULT_IMAGE_SIZE = 640


# =========================================================
# YOLO / CV SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# =========================================================
# REPORT SETTINGS
# =========================================================

MAX_DETECTIONS_PER_IMAGE = 100
REPORT_LANGUAGE = "english"


# =========================================================
# RANDOM SEED (for reproducibility)
# =========================================================

RANDOM_SEED = 42