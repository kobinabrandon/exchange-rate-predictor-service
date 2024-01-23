import pandas as pd 
from argparse import ArgumentParser
from typing import Optional, Callable

from xgboost import XGBRegressor  
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

from src.paths import MODELS_DIR
from src.logger import get_console_logger
from src.baseline_model import train_test_split
from src.hyperparameter_tuning import optimise_hyperparameters
from src.preprocessing import transform_ts_data_into_features_and_target


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
    
    # Set up a train-test split that will be used after the model has been optimised
    train_test_split(X=X, y=y)
    
    logger.info(f"Train sample size: {len(X_train)}")
    logger.info(f"Test sample size: {len(X_test)}")
    
    if tune_hyperparameters:
        
        # Optimise the hyperparameters
        logger.info("Finding optimal values of hyperparameters with cross-validation")
        
        best_preprocessing_hyperparemeters, best_model_hyperparameters = \
            optimise_hyperparameters(model_fn=model_fn, trials = trials, X=X_train,
                y = y_train, experiment=experiment
            )
            
        logger.info(f"Best hyperparameters from preprocessing: {best_preprocessing_hyperparemeters}")
        logger.info(f"Best model hyperparameters: {best_model_hyperparameters}")
        
        pipeline = make_pipeline(
            get_preprocessing_pipeline(**best_preprocessing_hyperparemeters),
            model_fn(**best_model_hyperparameters)
        )
        
        experiment.add_tag("Hyperparameter tuning")
        
        # Train the model
        logger.info("Fitting the model")
        pipeline.fit(X_train, y_train)
        
        # Make predictions, and compute the test error
        predictions = pipeline.predict(X_test)
        test_error = mean_absolute_error(y_test, predictions)
        logger.info(f"Test M.A.E: {test_error}")
        experiment.log_metrics({"Test M.A.E": test_error})
        
        logger.info("Saving model to disk")
        
        # Save model locally
        with open(MODELS_DIR/"model.pkl", "wb") as f:
            
            pickle.dump(pipeline, f)
        
        # Log model in CometML's model registry
        experiment.log_model(
            str(model_fn), str(MODELS_DIR/"model.pkl")
        )
        
    else:
        
        logger.info("Using default hyperparameters")
        
        pipeline = make_pipeline(
            get_preprocessing_pipeline(), 
            model_fn()
        )
        

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--model", type=str, default="lasso")
    parser.add_argument("--tune_hyperparameters", action="store_true")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--trials", type=int, default=10)
    
    args = parser.parse_args()
    
    logger.info("Generating features and targets")

    if args.sample_size is None:
        
        features, target = transform_ts_data_into_features_and_target()

    else:
        
        features = features.head(args.sample_size)
        target = target.head(args.sample_size)
        
    logger.info("Training the model")
    
    train(
        X=features,
        y=target,
        tune_hyperparameters=args.tune_hyperparameters,
        trials=args.trials
    )
    

    