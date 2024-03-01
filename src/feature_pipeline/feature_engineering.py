import pandas as pd 
import pandas_ta as ta
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import get_console_logger
from src.miscellaneous import get_closing_price_columns


logger = get_console_logger()

class RSI(BaseEstimator, TransformerMixin):
    
    """
    This class is primarily concerned with making its transform method
    an instrument for feature engineering. In that method, we will 
    apply the pandas_ta's RSI indicator to the closing rates for every
    day for the past month, creating a corresponding column for these 
    RSI values each time.
    """
    
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
        
        for col in get_closing_price_columns(data = X):
            
            X.insert(
                loc= X.shape[1],
                column=f"RSI_{col}",
                value= ta.rsi(X[col], length=self.rsi_length).fillna(50), 
                allow_duplicates=True
                )
        return X
    
    
class EMA(BaseEstimator, TransformerMixin):
    
    """
    This class is primarily concerned with making its transform method
    an instrument for feature engineering. In that method, we will 
    apply the pandas_ta's EMA indicator to the closing rates for every
    day for the past month, creating a corresponding column for these 
    EMA values each time.
    """
    
    def __init__(self, ema_length):
        
        self.ema_length = ema_length
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame|pd.Series] = None):
        
        return self

    def transform(self, X: pd.DataFrame):
        
        for col in get_closing_price_columns(data = X):
        
            X.insert(
                loc=X.shape[1],
                column=f"EMA_{col}",
                value=ta.ema(X[col], length = self.ema_length).fillna(50),
                allow_duplicates=True
            )
        
        return X


def get_percentage_change(
    X: pd.DataFrame, 
    days: int,
    base_currency: str = "GBP",
    target_currency: str = "GHS"
    ) -> pd.DataFrame:
    
    """ 
    This function calculates the percentage change in the closing rate 
    from the previous day to a specified previous day.
    
    Returns: a dataframe that includes the calculated percentage return
    """
    
    X.insert(
        loc=X.shape[1],
        column=f"percentage_change_between_yesterday_and_{days}_days_ago",
        value= 100*(
            X[f"Closing_rate_{base_currency}{target_currency}_1_day_ago"] - X[f"Closing_rate_{base_currency}{target_currency}_{days}_day_ago"]
        )/ X[f"Closing_rate_{base_currency}{target_currency}_{days}_day_ago"]
    )
        
    return X
