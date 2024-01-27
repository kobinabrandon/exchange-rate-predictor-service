
from typing import Optional
from pydantic import BaseModel


class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    your_param: Optional[str] = None # an example optional parameter


def predict(item, run_id, logger):
    item = Item(**item)

    ### ADD YOUR CODE HERE
    my_results = {"prediction": item.prompt, "your_optional_param": item.your_param}
    my_status_code = 200 # if you want to return some status code

    ### RETURN YOUR RESULTS
    return {"my_result": my_results, "status_code": my_status_code} # return your results
