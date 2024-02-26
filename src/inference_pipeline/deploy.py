
import fire 
from cerebrium import deploy, model_type 

from src.config import settings
from src.logger import get_console_logger
from src.model_registry_api import load_model_from_registry
from src.paths import MODELS_DIR


logger = get_console_logger(name="deployment")


def deploy(
  from_model_registry: bool = False
): 
  
  logger.info("Deploying model to Cerebrium")
  
  if from_model_registry:
    
    logger.info("Loading model from model registry...")
    
    load_model_from_registry(
      workspace=settings.comet_workspace,
      api_key=settings.comet_api_key,
      model_name=
    )