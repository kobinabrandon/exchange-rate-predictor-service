from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class Health(BaseModel):
  
  # Introduced to solve a namespace issue involving "model_"
  model_config = ConfigDict(protected_namespaces=())
  
  name: str
  api_version: str
  model_version: str


class PredictionResults(BaseModel):
  
  prediction: Optional[List[float]]
  
  
class Features(BaseModel):

  Closing_rate_GBPGHS_30_day_ago: float
  Closing_rate_GBPGHS_29_day_ago: float
  Closing_rate_GBPGHS_28_day_ago: float
  Closing_rate_GBPGHS_27_day_ago: float
  Closing_rate_GBPGHS_26_day_ago: float
  Closing_rate_GBPGHS_25_day_ago: float
  Closing_rate_GBPGHS_24_day_ago: float
  Closing_rate_GBPGHS_23_day_ago: float
  Closing_rate_GBPGHS_22_day_ago: float
  Closing_rate_GBPGHS_21_day_ago: float
  Closing_rate_GBPGHS_20_day_ago: float
  Closing_rate_GBPGHS_19_day_ago: float
  Closing_rate_GBPGHS_18_day_ago: float
  Closing_rate_GBPGHS_17_day_ago: float
  Closing_rate_GBPGHS_16_day_ago: float
  Closing_rate_GBPGHS_15_day_ago: float
  Closing_rate_GBPGHS_14_day_ago: float
  Closing_rate_GBPGHS_13_day_ago: float
  Closing_rate_GBPGHS_12_day_ago: float
  Closing_rate_GBPGHS_11_day_ago: float
  Closing_rate_GBPGHS_10_day_ago: float
  Closing_rate_GBPGHS_9_day_ago: float
  Closing_rate_GBPGHS_8_day_ago: float
  Closing_rate_GBPGHS_7_day_ago: float
  Closing_rate_GBPGHS_6_day_ago: float
  Closing_rate_GBPGHS_5_day_ago: float
  Closing_rate_GBPGHS_4_day_ago: float
  Closing_rate_GBPGHS_3_day_ago: float
  Closing_rate_GBPGHS_2_day_ago: float
  Closing_rate_GBPGHS_1_day_ago: float
 
    
class MultipleFeatureInputs(BaseModel):
  
  inputs: List[Features]
  
  class Config:
       
    json_schema_extra = {
      "example": {
        "inputs": [
          {
            'Closing_rate_GBPGHS_30_day_ago': 15.32124,
            'Closing_rate_GBPGHS_29_day_ago': 15.32124,
            'Closing_rate_GBPGHS_28_day_ago': 15.32124,
            'Closing_rate_GBPGHS_27_day_ago': 15.32124,
            'Closing_rate_GBPGHS_26_day_ago': 15.32124,
            'Closing_rate_GBPGHS_25_day_ago': 15.32124,
            'Closing_rate_GBPGHS_24_day_ago': 15.32124,
            'Closing_rate_GBPGHS_23_day_ago': 15.32124,
            'Closing_rate_GBPGHS_22_day_ago': 15.32124,
            'Closing_rate_GBPGHS_21_day_ago': 15.32124,
            'Closing_rate_GBPGHS_20_day_ago': 15.32124,
            'Closing_rate_GBPGHS_19_day_ago': 15.32124,
            'Closing_rate_GBPGHS_18_day_ago': 15.32124,
            'Closing_rate_GBPGHS_17_day_ago': 15.32124,
            'Closing_rate_GBPGHS_16_day_ago': 15.32124,
            'Closing_rate_GBPGHS_15_day_ago': 15.32124,
            'Closing_rate_GBPGHS_14_day_ago': 15.32124,
            'Closing_rate_GBPGHS_13_day_ago': 15.32124,
            'Closing_rate_GBPGHS_12_day_ago': 15.32124, 
            'Closing_rate_GBPGHS_11_day_ago': 15.32124,
            'Closing_rate_GBPGHS_10_day_ago': 15.32124,
            'Closing_rate_GBPGHS_9_day_ago': 15.32124,
            'Closing_rate_GBPGHS_8_day_ago': 15.32124,
            'Closing_rate_GBPGHS_7_day_ago': 15.32124,
            'Closing_rate_GBPGHS_6_day_ago': 15.32122,
            'Closing_rate_GBPGHS_5_day_ago': 15.31123,
            'Closing_rate_GBPGHS_4_day_ago': 15.30124,
            'Closing_rate_GBPGHS_3_day_ago': 15.23124,
            'Closing_rate_GBPGHS_2_day_ago': 15.1124,
            'Closing_rate_GBPGHS_1_day_ago': 15.32124
          }
        ]
      }
    }
    
    