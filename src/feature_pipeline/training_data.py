import pandas as pd 
import fire 

from src.feature_pipeline.data_extraction import update_ohlc
from src.feature_pipeline.data_transformations import transform_ts_data_into_features_and_target, get_preprocessing_pipeline


def make_training_data() -> pd.DataFrame:
  
  """
  Use all the functions created in the feature pipeline 
  so far to construct the training dataset.

  Returns:
      pd.DataFrame: the full training data
  """
  
  quotes = update_ohlc()

  features, target = transform_ts_data_into_features_and_target(original_data=quotes)
  pipe = get_preprocessing_pipeline()
  
  features = pipe.fit_transform(features)
  
  features["Closing_rate_next_day"] = target
  
  return features 

  
if __name__ == "__main__":
  
  make_training_data()
  