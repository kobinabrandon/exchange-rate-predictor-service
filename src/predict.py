from pydantic import BaseModel 
from src.logger import get_console_logger
from src.model_registry_api import load_model_from_registry


logger = get_console_logger("deployer")

try:
    
    from cerebrium import get_secret
    COMET_ML_WORKSPACE = get_secret("COMET_ML_WORKSPACE")
    COMET_ML_API_KEY = get_secret("COMET_ML_API_KEY")
    COMET_ML_MODEL_NAME = get_secret("COMET_ML_MODEL_NAME")
    
except ImportError:
    
    import os
    COMET_ML_WORKSPACE = os.environ["COMET_ML_WORKSPACE"]
    COMET_ML_API_KEY = os.environ["COMET_ML_API_KEY"]
    COMET_ML_MODEL_NAME = os.environ["COMET_ML_API_KEY"]    
    
    
model = load_model_from_registry(
    workspace=COMET_ML_WORKSPACE,
    api_key=COMET_ML_API_KEY,
    model_name=COMET_ML_MODEL_NAME
)
    
class Item(BaseModel):
    Closing_rate_(GBPGHS)_30_day_ago: float
    Closing_rate_(GBPGHS)_29_day_ago: float
    Closing_rate_(GBPGHS)_28_day_ago: float
    Closing_rate_(GBPGHS)_27_day_ago: float
    Closing_rate_(GBPGHS)_26_day_ago: float
    Closing_rate_(GBPGHS)_25_day_ago: float
    Closing_rate_(GBPGHS)_24_day_ago: float
    Closing_rate_(GBPGHS)_23_day_ago: float
    Closing_rate_(GBPGHS)_22_day_ago: float
    Closing_rate_(GBPGHS)_21_day_ago: float
    Closing_rate_(GBPGHS)_20_day_ago: float
    Closing_rate_(GBPGHS)_19_day_ago: float
    Closing_rate_(GBPGHS)_18_day_ago: float
    Closing_rate_(GBPGHS)_17_day_ago: float
    Closing_rate_(GBPGHS)_16_day_ago: float
    Closing_rate_(GBPGHS)_15_day_ago: float
    Closing_rate_(GBPGHS)_14_day_ago: float
    Closing_rate_(GBPGHS)_13_day_ago: float
    Closing_rate_(GBPGHS)_12_day_ago: float
    Closing_rate_(GBPGHS)_11_day_ago: float
    Closing_rate_(GBPGHS)_10_day_ago: float
    Closing_rate_(GBPGHS)_9_day_ago: float
    Closing_rate_(GBPGHS)_8_day_ago: float
    Closing_rate_(GBPGHS)_7_day_ago: float
    Closing_rate_(GBPGHS)_6_day_ago: float
    Closing_rate_(GBPGHS)_5_day_ago: float
    Closing_rate_(GBPGHS)_4_day_ago: float
    Closing_rate_(GBPGHS)_3_day_ago: float
    Closing_rate_(GBPGHS)_2_day_ago: float
    Closing_rate_(GBPGHS)_1_day_ago: float
    
    
def predict(item, run_id, logger):
    item = Item(**item)

    import pandas as pd 
    data = pd.DataFrame([item.dict()])
    
    prediction = model.predict(data)[0]
    
    return {"Prediction": prediction}
    