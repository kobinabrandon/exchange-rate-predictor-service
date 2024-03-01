import os
from pathlib import Path 

PARENT_DIR = Path(__file__).parent.resolve().parent
MODELS_DIR = PARENT_DIR/"models"
DATA_DIR = PARENT_DIR/"data"
TRAINING_DATA_DIR = DATA_DIR/"training"

RAW_DATA_DIR = DATA_DIR/"raw"
DAILY_DATA_DIR = RAW_DATA_DIR/"daily"



for folder in [MODELS_DIR, DATA_DIR, RAW_DATA_DIR, DAILY_DATA_DIR, TRAINING_DATA_DIR]:
    
    if not Path(folder).exists():
        os.mkdir(folder)
        