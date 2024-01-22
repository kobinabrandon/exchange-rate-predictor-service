import optuna
import pandas as pd
import numpy as np 

from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor

from comet_ml import Experiment

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from typing import Callable, Tuple, Dict

from src.logger import get_console_logger
from src.preprocessing import get_preprocessing_pipeline


logger = get_console_logger()

def sample_hyperparameters(
    model_fn: Callable,
    trial: Trial, 
) -> dict[str, str|int|float]:
    
    """
    Here we establish the range of values for each hyperparameter
    over which we will search.

    Raises:
        NotImplementedError: This error will be raised if the user
        enquires about a model that I have not yet investigated.

    Returns:
        dict[str, str|int|float]: the output will be a dictionary
        with strings as keys, and either a string, integer, or a
        float as the corresponding value. 
    """
    
    if model_fn == Lasso:
        
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True)
        }
        
    elif model_fn == LGBMRegressor:
        
        return {
            "metric": "mae",
            "verbose": -1,
            "num_leaves": trial.suggest_float("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "min_child_samples": trial.suggest_float("min_child_samples", 3, 100)
        }
        
    elif model_fn == XGBRegressor:
        
        return {
            "metric": "mae",
            "verbose": -1,
            "max_depth": trial.suggest_float("max_depth", 1.0, 10.0),
            "eta": trial.suggest_float("feature_fraction", 0.01, 0.3),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1.0),
            "subsample": trial.suggest_float("subsample", 0, 1.0)
        }
        
    else:
        raise NotImplementedError("This model is yet to be implemented")
    
    
def optimise_hyperparameters(
    model_fn: Callable,
    trials: int, 
    X: pd.DataFrame,
    y: pd.Series, 
    experiment: Experiment
) -> Tuple[Dict, Dict]:
    
    """
    We will be optimising two sets of hyperparameters: those that form part of
    our preprocessing operations, and those that control model performance.

    The optimal hyperparameters will be found by minimising the error function
    defined below.

    Returns:
        Tuple[Dict, Dict]: a tuple of dictionaries, where the first dictionary
        consists of the best values of the preprocessing hyperparameter, and 
        the second consists of the best values of the model's hyperparameter
    """
    
    assert model_fn in [Lasso, LGBMRegressor, XGBRegressor]
    
    def objective(trial: optuna.Trial.trial) -> float:
        
        """
        This is the error function that we will be minimising.

        Returns:
            float: the average score across all cross-validation splits.
        """
            
        hyperparameters_for_preprocessing = {
            "rsi_length": trial.suggest_int("rsi_length", 5, 20),
            "ema_length": trial.suggest_int("ema_length", 5, 30)
        }
        
        # Initiate the appropriate model hyperparameters
        model_hyperparameters = sample_hyperparameters(model_fn=model_fn, trial=trial)
        
        # Set up a time series split with 5 splits
        tss = TimeSeriesSplit(n_splits=5)
        scores = []
        
        logger.info(f"{trial.number=}")
        
        # Split the data, implement preprocessing, and instatiate the model using
        # the values of the shyperparameters selected by each trial.
        for split_number, (train_index, test_index) in enumerate(tss.split(X)):
            
            # Establish the training and test datasets
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            logger.info(f"{split_number=}")
            logger.info(f"{len(X_train)=}")
            logger.info(f"{len(X_val)=}")
            
            pipeline = make_pipeline(
                get_preprocessing_pipeline(**hyperparameters_for_preprocessing),
                model_fn(**model_hyperparameters)
            )
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores.append(mae)
            
            logger.info(f"{mae=}")
        
        # Compute the average of the accuracy scores
        average_score = np.array(scores).mean()
        
        return average_score 
    
    logger.info("Searching for hyperparameters")
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = trials)
    
    best_params = study.best_params
    best_value = study.best_value
    
    best_preprocessing_hyperparemeters = {
        key: value for key, value in best_params.items() if key.startswith("rsi") or key.startswith("ema")
    }
    
    best_model_hyperparameters = {
        key: value for key, value in best_params.items() if not key.startswith("rsi") and not key.startswith("ema")
    }
    
    logger.info("The best parameters are:")
    for key, value in best_params.items():
        
        logger.info(f"{key}: {value}")
        
    logger.info(f"Best MAE: {best_value}")    
    
    experiment.log_metric(f"Cross validation MAE: {best_value}")
    
    return best_preprocessing_hyperparemeters, best_model_hyperparameters
