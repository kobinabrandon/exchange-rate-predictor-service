import pandas as pd 
from typing import Optional, Callable

from xgboost import XGBRegressor  
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso

from src.logger import get_console_logger
from src.baseline_model import train_test_split


logger = get_console_logger()

def get_model(model: str) -> Callable:
    
    if model == "Lasso":
        
        return Lasso

    elif model == "xgboost":
        
        return XGBRegressor
    
    elif model == "lightgbm":
        
        return LGBMRegressor
    
    else:
        
        raise ValueError(f"The model {model} that you have requested is unknown.")


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparameters: Optional[bool] = True,
    hyperparameter_trials: Optional[int] = 10
) -> None:
    
    """
    Perform a train-test split using the features "X", and target "y", 
    and train a selected model, with or without prior hyperparameter
    tuning.
    
    Credit to Pau Labarta Bajo for nearly all the code in this module.
    """
    
    model_fn = get_model(model)
    
    # Log an experimental run of said model 
    experiment = Experiment(
        api_key=os.environ.get("COMET_ML_API_KEY"),
        workspace=os.environ.get("COMET_ML_WORKSPACE"),
        project_name = "exchange_range_predictor"
    )
     
    experiment.add_tag(model)
    
    train_test_split(X=X, y=y)
    logger.info(f"Train sample size: {len(X_train)}")
    logger.info(f"Test sample size: {len(X_test)}")
    
    if tune_hyperparameters:
        
        logger.info("Finding optimal values of hyperparameters with cross-validation")
        
        