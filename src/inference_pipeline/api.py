from typing import Any 

import pandas as pd 

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from src.config import settings
from src.logger import get_console_logger
from src.inference_pipeline.schemas import Health, Rates, PredictionResults, MultiplePastClosingRateInputs


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
  input_data:MultiplePastClosingRateInputs, 
  status_code=200
  ) -> Any:

  input_data = pd.DataFrame(
    jsonable_encoder(input_data.inputs)
  )
  
  logger.info(f"Making predictions on inputs: {input_data.inputs}")
  
  