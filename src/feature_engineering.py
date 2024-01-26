import pandas as pd 
import pandas_ta as ta
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from src.logger import get_console_logger
from src.miscellaneous import get_closing_price_columns


logger = get_console_logger()

class RSI(BaseEstimator, TransformerMixin):
    
    def __init__(self, rsi_length: int = 14):
        
        self.rsi_length = rsi_length
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame|pd.Series] = None):
        
        return self 

    def transform(self, X: pd.DataFrame):
        
        """
        Compute RSI for the closing rates(for each column) in the feature dataset.

        Returns:
            X: this dataframe will consist of the features.
        """
        
        logger.info("Adding RSI to the features")
        
        for col in get_closing_price_columns(X = X):
            
            X.insert(
                loc= X.shape[1],
                column=f"RSI derived from {col}",
                value= ta.rsi(X[f"{col}"], length=self.rsi_length).fillna(50), 
                allow_duplicates=True
                )
        return X
    
    
class EMA(BaseEstimator, TransformerMixin):
    
    def __init__(self, ema_length):
        
        self.ema_length = ema_length
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame|pd.Series] = None):
        
        return self

    def transform(self, X: pd.DataFrame):
        
        for col in get_closing_price_columns(X = X):
        
            X.insert(
                loc=X.shape[1],
                column=f"EMA derived from {col}",
                value=ta.ema(X[f"{col}"], length = self.ema_length),
                allow_duplicates=True
            )
        
        return X


def get_percentage_return(X: pd.DataFrame, days: int) -> pd.DataFrame:
    
    """ """

    X[f"percentage_return_{days}_day"] = \
        (
            X[f"Closing rate_(GBPGHS)_1_day_ago"] - X[f"Closing rate_(GBPGHS)_{days}_day_ago"]
        )/ X[f"Closing rate_(GBPGHS)_{days}_day_ago"]
        
    return X
