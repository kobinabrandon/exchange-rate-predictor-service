import pandas as pd 

from sklearn.metrics import mean_absolute_error

from src.logger import get_console_logger
from src.feature_pipeline.data_extraction import update_ohlc
from src.feature_pipeline.data_transformations import transform_ts_data_into_features_and_target


logger = get_console_logger() 

def train_baseline(
    X: pd.DataFrame, 
    y: pd.Series, 
    base_currency: str = "GBP", 
    target_currency: str = "GHS"
    ) -> None:
    
    """
    Fit a rudimentary model that predicts the closing rate 
    will be the same as yesterday's  closing rate, and 
    deliver the mean absolute error on the test data.
    """
    
    train_sample_size = int(0.9*len(X))
    X_train, X_test = X[:train_sample_size], X[train_sample_size:] # noqa F841
    y_train, y_test = y[:train_sample_size], y[train_sample_size:] # noqa: F841
    
    # Evaluate the performance of this baseline model
    baseline_predictions = X_test[f"Closing_rate_{base_currency}{target_currency}_1_day_ago"]
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    
    logger.info(f"Test M.A.E: {baseline_mae}")
    
    
if __name__ == "__main__":
    
    logger.info("Generating features and targets")
    features, target = transform_ts_data_into_features_and_target(original_data=update_ohlc())
    
    logger.info("Starting training")
    train_baseline(X = features, y = target)
    