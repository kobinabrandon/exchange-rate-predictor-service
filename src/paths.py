import os
from pathlib import Path 

PARENT_DIR = Path(__file__).parent.resolve().parent
MODELS_DIR = PARENT_DIR/"models"
DATA_DIR = PARENT_DIR/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
DAILY_DATA_DIR = RAW_DATA_DIR/"daily"
HOURLY_DATA_DIR = RAW_DATA_DIR/"hourly"

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)
    