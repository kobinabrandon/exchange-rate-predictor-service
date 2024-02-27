from typing import Any, Callable

import pandas as pd 

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from sklearn.linear_model import Lasso 
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import settings
from src.logger import get_console_logger
from src.inference_pipeline.schemas import Health, Rates, PredictionResults, MultiplePastClosingRateInputs
from src.inference_pipeline.model_registry_api import load_model_from_registry


logger = get_console_logger()
api_router = APIRouter()


@api_router.get(path="/health", response_model=Health, status_code=200)
def health() -> dict:
  
  health = Health(
    name=settings.comet_project_name,
    api_version=settings.api_version,
    model_version=settings.modelversion
  )
  
  return health.dict()


@api_router.post(path="/predict", response_model=PredictionResults, status_code=200)
async def predict(
  input_data: MultiplePastClosingRateInputs, 
  model: str,
  status_code=200,
  from_model_registry: bool = False
  ) -> Any:

  input_data = pd.DataFrame(
    jsonable_encoder(input_data.inputs)
  )
  
  logger.info(f"Making predictions on inputs: {input_data.inputs}")
  
  models_and_names = {
    "lasso": Lasso,
    "Lasso": Lasso,
    "xgboost": XGBRegressor,
    "lightbgm": LGBMRegressor
  }
  
  if from_model_registry:
    
    if model in models_and_names.keys():
    
      load_model_from_registry(
        workspace=settings.comet_workspace,
        api_key=settings.comet_api_key,
        model_name=,
        status="Production" 
      )
  