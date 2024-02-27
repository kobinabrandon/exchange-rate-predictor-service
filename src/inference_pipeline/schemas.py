from datetime import date, datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class Health(BaseModel):
  
  model_config = ConfigDict(
    protected_namespaces=()
  )
  
  name: str
  api_version: str
  model_version: str


class PredictionResults(BaseModel):
  version: str
  predictions: Optional[List[float]]
  
  
class Rates(BaseModel):
    Date: date
    Opening_rate_GHSGBP: float
    Peak_rate_GHSGBP: float 
    Lowest_rate_GHSGBP: float 
    Closing_rate_GHSGBP: float
    
    
class MultiplePastClosingRateInputs(BaseModel):
  
  inputs : List[Rates]
  
  class Config:
       
    json_schema_extra = {
      "example": {
        "inputs": [
          {
            "Date": datetime(2024,3,3),
            "Opening_rate_GHSGBP": 15.123456,
            "Peak_rate_GHSGBP": 15.213456, 
            "Lowest_rate_GHSGBP": 15.102156, 
            "Closing_rate_GHSGBP": 15.173456
          }
        ]
      }
    }