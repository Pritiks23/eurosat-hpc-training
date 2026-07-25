
from pathlib import Path
import torch


# ============================
# Project directories
# ============================

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

METRIC_DIR = ROOT_DIR / "metrics"
METRIC_DIR.mkdir(exist_ok=True)

RUN_DIR = ROOT_DIR / "runs"
RUN_DIR.mkdir(exist_ok=True)


# ============================
# Dataset configuration
# ============================

NUM_CLASSES = 10

IMAGE_SIZE = 64


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15



# ============================
# Training configuration
# ============================

SEED = 42

EPOCHS = 20

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ============================
# DataLoader
# ============================

NUM_WORKERS = 8

PIN_MEMORY = True



# ============================
# Hardware
# ============================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


USE_AMP = True



# ============================
# Image normalization
# ============================

NORMALIZE_MEAN = 0.5

NORMALIZE_STD = 0.2



# ============================
# Checkpoints
# ============================

BEST_MODEL_PATH = (
    MODEL_DIR /
    "best_model.pth"
)


LAST_CHECKPOINT_PATH = (
    MODEL_DIR /
    "last_checkpoint.pth"
)



# ============================
# CUDA extension
# ============================

USE_CUSTOM_CUDA = True


CUDA_MEAN = 0.5

CUDA_STD = 0.2
