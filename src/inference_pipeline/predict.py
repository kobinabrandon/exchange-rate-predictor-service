from pydantic import BaseModel 
from src.logger import get_console_logger
from src.model_registry_api import load_model_from_registry

from src.inference_pipeline.model_serving.schemas import PastClosingRates


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
    
def predict(item) -> dict:
    
    item = PastClosingRates(**item)

    import pandas as pd 
    
    data = pd.DataFrame([item.dict()])
    
    prediction = model.predict(data)[0]
    
    return {"Prediction": prediction}



