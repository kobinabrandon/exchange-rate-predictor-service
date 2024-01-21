import pandas as pd 
import pandas_ta as ta
from typing import Optional
from src.logger import get_console_logger

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

logger = get_console_logger()


class RSI(BaseEstimator, TransformerMixin):
    
    def __init__(self, length: int = 14):
        
        self.length = length
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame|pd.Series] = None):
        
        return self 

    def transform(self, X: pd.DataFrame):
        

        if f"Closing rate (GBPGHS)" in X.columns:

            X[f"RSI (GBPGHS)"] = ta.rsi(X[f"Closing rate (GBPGHS)"], length=length).fillna(50)

        else:

            logger.debug("Check the currencies you have provided for typos")
            
        return X
    
    
class EMA(BaseEstimator, TransformerMixin):
    
    def __init__(self, length = 14):
        
        self.length = length
        
    def fit(X: pd.DataFrame, y: Optional[pd.DataFrame|pd.Series] = None):
        
        return X

    def transform(self, X: pd.DataFrame):
        
        X["EMA (GBPGHS)"] = ta.ema(X["Closing rate (GBPGHS)"], length = 14).fillna()


def get_percentage_return(
    X: pd.DataFrame, 
    target_currency: str,
    days: int
) -> pd.DataFrame:
    
    """ """

    X[f"percentage_return_{days}_day"] = \
        (
            X[f"Closing rate(GHS)_1_day_ago"] - X[f"Closing rate(GHS)_{days}_day_ago"]
        )/ X[f"Closing rate(GHS)_{days}_day_ago"]
        
    return X
