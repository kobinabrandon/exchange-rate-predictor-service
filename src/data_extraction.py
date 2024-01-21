import os 
import requests         
import pandas as pd   
from tqdm import tqdm         
from pathlib import Path

from datetime import datetime
from src.paths import DAILY_DATA_DIR

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")


def get_api_response(date: datetime) -> dict:
    
    """
    We fetch the API response.

    Returns:
        date (datetime): The date with respect to which 
                         want data.
    """
    
    URL = f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/fx/{date.strftime('%Y-%m-%d')}?adjusted=true&apiKey={POLYGON_API_KEY}"

    endpoint = requests.get(URL)

    if endpoint.status_code != 200:
        
        print(f"Error {endpoint.status_code} - {endpoint.text}")

    else:

        return endpoint.json()
            

def extract_results(
    response: dict, 
    date: datetime,
    index: int) -> pd.DataFrame:
    
    """ 
    Here we search through the value associated with the "results"
    key in the API response, parse it for the  OHLC data, and put
    it all into a dataframe.

    Returns:
        pd.DataFrame: all the OHLC data from the given start date 
                      to the end date.
    """
    
    if "results" in response.keys():

        for results in response["results"]:

            if results["T"] == f"C:GBPGHS":
                
                opening_rate = results["o"]
                peak_rate = results["h"]
                lowest_rate = results["l"]
                closing_rate = results["c"]
                
                return pd.DataFrame(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        f"Opening rate (GBPGHS)": opening_rate,
                        f"Peak rate (GBPGHS)": peak_rate,
                        f"Lowest rate (GBPGHS)": lowest_rate,
                        f"Closing rate (GBPGHS)": closing_rate
                    }, index = [index]
                )
    

def get_daily_ohlc(start_date: datetime, end_date: datetime) -> pd.DataFrame:

    """
    Download currency exchange rate data with pounds sterling(GBP)
    and the Ghanaian cedi (GHS) as the base and target 
    currencies respectively.
    
    Returns:
        pd.DataFrame: a dataframe of daily exchange rates during the specified period               
    """
    
    dataframe = pd.DataFrame()
    file_name = DAILY_DATA_DIR/f"GBPGHS_{start_date}_{end_date}.parquet"

    if file_name.exists():

        print("That file already exists")

        dataframe = pd.read_parquet(file_name)

        return dataframe

    else:

        dataframe = pd.DataFrame()
        index = 0

        date_range = pd.date_range(
            start=start_date,
            end=end_date
        )

        for date in tqdm(date_range):
            
            # The forex market closes on Saturdays (which datetime sees as day 5)
            if datetime.weekday(date) != 5:
            
                api_response = get_api_response(date=date)
                
                current_date_data = extract_results(response=api_response, date=date, index=index)

                dataframe = pd.concat([dataframe, current_date_data])
                index += 1
        
        dataframe = dataframe.reset_index(drop = True) 
        dataframe.to_parquet(path=DAILY_DATA_DIR/f"GBPGHS_{start_date}_{end_date}.parquet")

        return dataframe


def update_ohlc(initial_data: pd.DataFrame) -> pd.DataFrame:
    
    """
    Suppose that we have a dataframe that we have already 
    downloaded. As time goes on, we would like to update
    it. For each date between the start and the current 
    time, we will make a dataframe of OHLC data, and 
    concatenate it with the input dataframe.

    Returns:
        pd.DataFrame: This is an updated dataframe
    """
    
    initial_start_date = datetime.strptime(initial_data["Date"].iloc[0], "%Y-%m-%d")
    update_from = datetime.strptime(initial_data["Date"].iloc[-1], "%Y-%m-%d")
    
    date_range = pd.date_range(start=update_from, end=datetime.utcnow())
    
    index = initial_data.index[-1]
    
    for date in tqdm(date_range):
        
        download = get_api_response(date=date) 
        new_data = extract_results(response=download, date = date, index = index)

        dataframe = pd.concat([initial_data, new_data])
        index += 1

    dataframe = dataframe.reset_index(drop=True)
    dataframe.to_parquet(path=DAILY_DATA_DIR/f"GBPGHS_{initial_start_date}_{datetime.utcnow()}.parquet")

    return dataframe

