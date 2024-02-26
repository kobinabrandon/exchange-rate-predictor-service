import pandas as pd 

from typing import Tuple, Optional
from pydantic import BaseModel, ValidationError


def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
  
  for col in input_data.columns:
    
    if input_data[col].isna().sum() == 0:
      
      pass 
    
    else:
      
      raise ValidationError()  


#class RateDataInputScheme(BaseModel): 