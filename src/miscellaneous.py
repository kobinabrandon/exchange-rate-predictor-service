import pandas as pd 
from typing import List

def get_closing_price_columns(data: pd.DataFrame) -> List[str]:
    
    """Get  a list of the column names in the feature set that contain the closing rates."""

    return [
        column for column in data.columns if column.startswith("Closing_rate" )
    ]


def get_subset_of_features(X: pd.DataFrame) -> pd.DataFrame:
    
    subset = ["Closing_rate_GBPGHS_1_day_ago", "percentage_return_2_day", "percentage_return_30_day"]
    
    return X[
        subset + [col for col in X.columns if col.startswith("RSI") and col.startswith("EMA")]
    ]
