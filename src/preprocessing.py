import numpy as np
import pandas as pd
from typing import Optional
from fire import Fire

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.feature_engineering import get_percentage_return, RSI, EMA


def get_cutoff_indices(
        data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
) -> list:
    
    """
    This function will take a dataframe and will look at
    the first input_seq_len rows+1, and then records a tuple 
    consisting of:

    - the first index
    - index number "input_seq_len"
    - the last index.

    Returns:
        list: returns a list of tuples of these indices
    """

    indices = []
    stop_position = len(data) - 1

    subseq_first_index = 0
    subseq_mid_index = input_seq_len
    subseq_last_index = input_seq_len + 1

    while subseq_last_index <= stop_position:
        
        indices.append(
            (subseq_first_index, subseq_mid_index, subseq_last_index)
        )
        
        subseq_first_index += step_size
        subseq_mid_index += step_size
        subseq_last_index += step_size

    return indices


def transform_ts_data_into_features_and_target(
        original_data: pd.DataFrame,
        input_seq_len: Optional[int] = 7,
        step_size: Optional[int] = 1
    ) -> list:
    
    """
    This transforms our time series data into a feature-target
    format that is conducive for training supervised training
    algorithms.

    Returns:
        tuple: consisting of the dataframe of features, and
               a pandas series of the target variable.
    """

    ts_data = original_data[
        ["Date", f"Closing rate (GBPGHS)"]
    ]

    ts_data = ts_data.sort_values(by=["Date"])

    indices = get_cutoff_indices(data=ts_data, input_seq_len=input_seq_len, step_size=step_size)

    x = np.ndarray(
        shape=(len(indices), input_seq_len), dtype=np.float32
    )

    y = np.ndarray(
        shape=(len(indices)), dtype=np.float32
    )
    
    dates = []
    for i, idx in enumerate(indices):
        
        x[i,:] = ts_data.iloc[idx[0]: idx[1]]["Closing rate (GBPGHS)"].values
        y[i] = ts_data.iloc[idx[1]: idx[2]]["Closing rate (GBPGHS)"].values
        dates.append(ts_data.iloc[idx[1]]["Date"])

    features = pd.DataFrame(
        x, columns=[
            f"Closing rate (GBPGHS) {i + 1} day ago" for i in reversed(range(input_seq_len))
        ]
    )

    targets = pd.DataFrame(y, columns=[f"Closing rate (GBPGHS) next day"])

    return features, targets[f"Closing rate (GBPGHS) next day"]


def get_closing_price_columns(X: pd.DataFrame) -> list[str]:
    
    """Get  a list of the column names in the feature set that contain the closing rates."""

    return [column for column in X.columns if "Closing rate" in column]


def get_subset_of_features(X: pd.DataFrame) -> pd.DataFrame:

    return X[
        [f"Closing rate(GBPGHS)_1_day_ago", "percentage_return_2_day", "percentage_return_30_day",
         f"RSI(GBP)"]
    ]


def get_preprocessing_pipeline(length: int) -> Pipeline:
    
    """ Returns a pipeline that combines all of the preprocessing steps """

    return make_pipeline(

        FunctionTransformer(func=get_percentage_return, kw_args={"days": 2}),
        FunctionTransformer(func=get_percentage_return, kw_args={"day": 30}),

        RSI(length=rsi_length),
        EMA(length=ema_length),

        FunctionTransformer(func=get_subset_of_features)
    )


if __name__ == "__main__":
    
    features, target = Fire(transform_ts_data_into_features_and_target)

    preprocessing_pipeline = get_preprocessing_pipeline(length=14)

    get_preprocessing_pipeline.fit(features)

    X = preprocessing_pipeline.transform(features)

    print(X.head())
