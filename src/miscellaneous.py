import pandas as pd 

def get_closing_price_columns(X: pd.DataFrame) -> list[str]:
    
    """Get  a list of the column names in the feature set that contain the closing rates."""

    return [column for column in X.columns if "Closing rate" in column]


def get_subset_of_features(X: pd.DataFrame) -> pd.DataFrame:
    
    subset = [f"Closing rate (GBPGHS) 1 day ago", "percentage_return_2_day", "percentage_return_30_day"]
    
    return X[
        subset + [col for col in X.columns if col.startswith("RSI")]
    ]
