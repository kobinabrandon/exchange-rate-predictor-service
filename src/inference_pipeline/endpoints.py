import pickle 
import pandas as pd 

from typing import Any

from comet_ml.exceptions import CometRestApiException
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

from sklearn.linear_model import Lasso 
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import settings
from src.paths import MODELS_DIR
from src.logger import get_console_logger

from src.feature_pipeline.data_transformations import transform_ts_data_into_features_and_target, get_preprocessing_pipeline
from src.inference_pipeline.schemas import Health, Features, PredictionResults, MultipleFeatureInputs
from src.inference_pipeline.model_registry import load_model_from_registry


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
  input_data: MultipleFeatureInputs, 
  model: str,
  status_code=200,
  from_model_registry: bool = False
  ) -> Any:

  input_data = pd.DataFrame(
    jsonable_encoder(input_data.inputs)
  )
  
  #pipe = get_preprocessing_pipeline()
  #features = pipe.transform(features)
  
  logger.info("Making predictions on inputs:")
  
  models_and_names = {
    "lasso": Lasso,
    "Lasso": Lasso,
    "xgboost": XGBRegressor,
    "lightbgm": LGBMRegressor
  }
  
  if from_model_registry:
    
    if model in models_and_names.keys():
    
      try:
        
        logger.info("Loading model from model registry...")
        
        model = load_model_from_registry(
          workspace=settings.comet_workspace,
          api_key=settings.comet_api_key,
          model_name=model,
          status="Production" 
        )
        
        logger.info("Making predictions on inputs")
        
        prediction = model.predict(input_data)
        
        logger.info(f"Prediction: {prediction}")

        return PredictionResults(prediction=prediction)
        
      except CometRestApiException: 
        
        logger.error("This model has not been logged on the CometML model registry")
        quit()

    else:
      
      raise NotImplementedError("That model has not been implemented")
  
  else:
    
    logger.info("Deploying model from local pickle file...")
    
    try:
      
      with open(file=MODELS_DIR/f"Tuned {model} model.pkl", mode="rb") as saved_pkl:
      
        loaded_model = pickle.load(file=saved_pkl)

        logger.info("Making predictions on inputs")
        
        prediction = loaded_model.predict(input_data)
        
        logger.info(f"Predictions: {prediction}") 
        
        return PredictionResults(prediction=prediction)
      
    except FileNotFoundError as no_file:
      
      logger.error(no_file)
      