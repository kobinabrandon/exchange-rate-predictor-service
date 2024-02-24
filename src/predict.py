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
    Closing_rate_GBPGHS_30_day_ago: float
    Closing_rate_GBPGHS_29_day_ago: float
    Closing_rate_GBPGHS_28_day_ago: float
    Closing_rate_GBPGHS_27_day_ago: float
    Closing_rate_GBPGHS_26_day_ago: float
    Closing_rate_GBPGHS_25_day_ago: float
    Closing_rate_GBPGHS_24_day_ago: float
    Closing_rate_GBPGHS_23_day_ago: float
    Closing_rate_GBPGHS_22_day_ago: float
    Closing_rate_GBPGHS_21_day_ago: float
    Closing_rate_GBPGHS_20_day_ago: float
    Closing_rate_GBPGHS_19_day_ago: float
    Closing_rate_GBPGHS_18_day_ago: float
    Closing_rate_GBPGHS_17_day_ago: float
    Closing_rate_GBPGHS_16_day_ago: float
    Closing_rate_GBPGHS_15_day_ago: float
    Closing_rate_GBPGHS_14_day_ago: float
    Closing_rate_GBPGHS_13_day_ago: float
    Closing_rate_GBPGHS_12_day_ago: float
    Closing_rate_GBPGHS_11_day_ago: float
    Closing_rate_GBPGHS_10_day_ago: float
    Closing_rate_GBPGHS_9_day_ago: float
    Closing_rate_GBPGHS_8_day_ago: float
    Closing_rate_GBPGHS_7_day_ago: float
    Closing_rate_GBPGHS_6_day_ago: float
    Closing_rate_GBPGHS_5_day_ago: float
    Closing_rate_GBPGHS_4_day_ago: float
    Closing_rate_GBPGHS_3_day_ago: float
    Closing_rate_GBPGHS_2_day_ago: float
    Closing_rate_GBPGHS_1_day_ago: float
    
    
def predict(item, logger):
    
    item = Item(**item)

    import pandas as pd 
    
    data = pd.DataFrame([item.dict()])
    
    prediction = model.predict(data)[0]
    
    return {"Prediction": prediction}
    