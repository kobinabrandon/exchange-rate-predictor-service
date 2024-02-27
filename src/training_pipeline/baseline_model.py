import pandas as pd 

from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error

from src.config import settings
from src.logger import get_console_logger
from src.preprocessing import transform_ts_data_into_features_and_target


logger = get_console_logger() 

def train(
    X: pd.DataFrame, 
    y: pd.Series, 
    base_currency: str = "GBP", 
    target_currency: str = "GHS") -> None:
    
    """
    Log a CometML experiment which is a rudimentary model that
    predicts the closing rate will be the same as yesterday's 
    closing rate.
    """
    
    # Log the experiment with CometML
    experiment = Experiment(
        api_key=settings.comet_api_key,
        workspace=settings.comet_workspace,
        project_name = settings.comet_project_name
    )
    
    experiment.add_tag("baseline_model")
    
    train_sample_size = int(0.9*len(X))
    X_train, X_test = X[:train_sample_size], X[train_sample_size:]
    y_train, y_test = y[:train_sample_size], y[train_sample_size:] # noqa: F841
    
    # Evaluate the performance of this baseline model
    baseline_predictions = X_test[f"Closing rate_({base_currency}{target_currency})_1_day_ago"]
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    
    logger.info(f"Train sample size: {len(X_train)}")
    logger.info(f"Test sample size: {len(X_test)}")
    logger.info(f"Test M.A.E: {baseline_mae}")
    
    # Log the M.A.E of the baseline model with CometML
    experiment.log_metrics({"Test M.A.E": baseline_mae})
    
    
if __name__ == "__main__":
    
    logger.info("Generating features and targets")
    features, target = transform_ts_data_into_features_and_target()
    
    logger.info("Starting training")
    train(X = features, y = target)
    