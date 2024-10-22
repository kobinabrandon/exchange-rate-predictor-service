import pickle
from typing import Optional, Callable

import pandas as pd 
from argparse import ArgumentParser

from comet_ml import Experiment

from xgboost import XGBRegressor  
from lightgbm import LGBMRegressor

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from src.config import settings
from src.paths import MODELS_DIR
from src.logger import get_console_logger
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparameters
from src.feature_pipeline.data_transformations import transform_ts_data_into_features_and_target, get_preprocessing_pipeline
from src.feature_pipeline.data_extraction import update_ohlc


logger = get_console_logger()

models_and_names = {
    "lasso": Lasso,
    "Lasso": Lasso,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor
}


def get_model(model: str) -> Callable:
    
    """
    Provide a way to invoke a specific model class
    with an appropriate string.

    Raises:
        NotImplementedError: indicates that the requested model
                             has not been implemented.

    Returns:
        Callable: the class of the requested model.
    """
    
    if model in models_and_names.keys():
        
        return models_and_names[model]
    
    else:
        
        raise NotImplementedError("The model that you have requested has not been implemented.")


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparameters: Optional[bool] = True,
    tuning_trials: Optional[int] = 10
) -> None:
    
    """
    Perform a train-test split using the features "X", and target "y", 
    and train a selected model, with or without prior hyperparameter
    tuning.
    
    Then log 
    
    Credit to Pau Labarta Bajo for nearly all the code in this module.
    
    """
    
    model_fn = get_model(model)
    
    # Log an experimental run of said model 
    experiment = Experiment(
        api_key=settings.comet_api_key,
        workspace=settings.comet_workspace,
        project_name = settings.comet_project_name
    )
     
    experiment.add_tag(model)
    
    # Set up a train-test split that will be used after the model has been optimised
    train_sample_size = int(0.9*len(X))
    X_train, X_test = X[:train_sample_size], X[train_sample_size:]
    y_train, y_test = y[:train_sample_size], y[train_sample_size:]
    
    logger.info(f"Train sample size: {len(X_train)}")
    logger.info(f"Test sample size: {len(X_test)}")
    
    if tune_hyperparameters:
        
        # Optimise the hyperparameters
        logger.info("Finding optimal values of hyperparameters with cross-validation")
        
        best_preprocessing_hyperparameters, best_model_hyperparameters = \
            optimise_hyperparameters(
                model_fn=model_fn, 
                tuning_trials = tuning_trials, 
                X=X_train,
                y = y_train, 
                experiment=experiment
            )
            
        logger.info(f"Best hyperparameters from preprocessing: {best_preprocessing_hyperparameters}")
        logger.info(f"Best model hyperparameters: {best_model_hyperparameters}")
        
        pipeline = make_pipeline(
            get_preprocessing_pipeline(**best_preprocessing_hyperparameters),
            model_fn(**best_model_hyperparameters)
        )
        
        experiment.add_tag("Tuned")
        
        # Train the model
        logger.info("Fitting the model")
        pipeline.fit(X_train, y_train)
        
        # Make predictions, and compute the test error
        predictions = pipeline.predict(X_test)
        test_error = mean_absolute_error(y_test, predictions)
        
        logger.info(f"Test M.A.E: {test_error}")
        experiment.log_metrics({"Test M.A.E": test_error})
        
        logger.info(f"Saving tuned {model} model to disk")
        
        # Save model locally
        with open(MODELS_DIR/f"Tuned {model} model.pkl", "wb") as f:
            
            pickle.dump(pipeline, f)
        
        # Log model in CometML's model registry
        experiment.log_model(
            model, str(MODELS_DIR/f"Tuned {model} model.pkl")
        )
        
    else:
        
        logger.info("Training an untuned model")
        
        pipeline = make_pipeline(
            get_preprocessing_pipeline(), 
            model_fn()
        )
        
        experiment.add_tag("Untuned")
        
        with open(MODELS_DIR/f"Untuned {model} model", "wb") as f:
            
            pickle.dump(pipeline, f)
        

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    # Lasso has been selected as the default model because it has performed best 
    parser.add_argument("--model", type=str, default="lasso")
    parser.add_argument("--tune_hyperparameters", action="store_true", default=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--tuning_trials", type=int, default=10)
    
    args = parser.parse_args()
    
    logger.info("Generating features and targets")
    
    features, target = transform_ts_data_into_features_and_target(
            original_data=update_ohlc()
        )

    if args.sample_size is not None:
        
        features = features.head(args.sample_size)
        target = target.head(args.sample_size)
        
    logger.info("Training the model")
    
    train(
        X=features,
        y=target,
        model=args.model,
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_trials=args.tuning_trials
    )
