import os 
import requests      
   
import pandas as pd   
from tqdm import tqdm         

from datetime import datetime
import fire

from src.config import settings
from src.paths import DAILY_DATA_DIR
from src.logger import get_console_logger


logger = get_console_logger()
POLYGON_API_KEY = settings.polygon_api_key


def get_api_response(date: datetime) -> dict:
    
    """
    We fetch the Polygon Forex API response.

    Returns:
        date (datetime): The date with respect to which 
                         want data.
    """
    
    URL = f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/fx/{date.strftime('%Y-%m-%d')}?adjusted\=true&apiKey={POLYGON_API_KEY}"

    endpoint = requests.get(URL)

    if endpoint.status_code != 200:
        
        print(f"Error {endpoint.status_code} - {endpoint.text}")

    else:

        return endpoint.json()
            

def extract_results(
    response: dict, 
    date: datetime,
    index: int
    ) -> pd.DataFrame:
    
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

            if results["T"] == f"C:{base_currency}{target_currency}":
                
                opening_rate = results["o"]
                peak_rate = results["h"]
                lowest_rate = results["l"]
                closing_rate = results["c"]
                
                return pd.DataFrame(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        f"Opening rate ({base_currency}{target_currency})": opening_rate,
                        f"Peak rate ({base_currency}{target_currency})": peak_rate,
                        f"Lowest rate ({base_currency}{target_currency})": lowest_rate,
                        f"Closing rate ({base_currency}{target_currency})": closing_rate
                    }, index = [index]
                )
    

def get_daily_ohlc(
    start_date: datetime = datetime(2017,1,1), 
    end_date: datetime = datetime.utcnow(),
    base_currency: str = "GBP",
    target_currency: str = "GHS"
    ) -> pd.DataFrame:

    """
    By default, download currency exchange rate data with 
    pounds sterling and the Ghanaian cedi as the base and 
    target currencies respectively.
    
    Args:
        start_date: the date from which we want to collect data
        end_date: the last date on which we want data
    
    Returns:
        pd.DataFrame: a dataframe of daily exchange rates during the specified period               
    """
    
    dataframe = pd.DataFrame()
    file_path = DAILY_DATA_DIR/f"{base_currency}{target_currency}_{start_date}_{end_date}.parquet"

    if file_path.exists():

        print("That file already exists")

        dataframe = pd.read_parquet(file_path)

        return dataframe

    else:

        index = 0
        dataframe = pd.DataFrame()

        date_range = pd.date_range(
            start=start_date,
            end=end_date
        )

        for date in tqdm(date_range):
            
            # The forex market closes on Saturdays (which the datetime package sees as day 5)
            if datetime.weekday(date) != 5:
            
                api_response = get_api_response(date=date)
                
                current_date_data = extract_results(
                    response=api_response, 
                    date=date, 
                    index=index
                    )

                dataframe = pd.concat(
                    [dataframe, current_date_data]
                )
                
                index += 1
        
        dataframe = dataframe.reset_index(drop = True) 
        dataframe.to_parquet(path=file_path)

        return dataframe
    

def get_newest_local_dataset() -> pd.DataFrame:
    
    """
    Returns the most recent file locally saved data file as a Pandas
    dataframe.
    
    Checks for the presence of any data in the folder where daily data 
    is kept. If one or more files are present, it returns the newsest 
    file (chronologically, not content-wise). If there is no saved 
    data, the function will download data using the default parameters.
    
    The primary purpose of this function is to provide a way for the 
    most up-to-date dataset to be used to generate training data.

    Returns:
        pd.DataFrame
    """
    
    with os.scandir(f"{DAILY_DATA_DIR}") as data:
        
        if any(data):
            
            import glob
            
            files = glob.glob(f"{DAILY_DATA_DIR}")
            
            newest_file = max(files, key=os.path.getctime)
            
            dataframe = pd.read_parquet(newest_file)
            
        else:
            
            dataframe = get_daily_ohlc()
        
    return dataframe



def update_ohlc() -> pd.DataFrame:
    
    """
    This function checks for an existing dataframe, and updates it.
    If it finds, it will update. If it doesn't find it, it will 
    default to fetching data from the beginning of 2017 till date.

    In the former case, for each date between the final date in the dataframe
    to the current date, we will make a dataframe of OHLC data, and 
    concatenate it with the input dataframe.

    Returns:return initial_data
        pd.DataFrame: This is the updated dataframe
    """
    
    logger.info("Checking the daily data folder for pre-existing files")
    
    with os.scandir(DAILY_DATA_DIR) as data:
        
        if any(data):
            
            logger.info("Found a file -> Let's update it")
            initial_data = get_newest_local_dataset()

            try:
                
                initial_start_date = datetime.strptime(initial_data["Date"].iloc[0], "%Y-%m-%d")
                update_from = datetime.strptime(initial_data["Date"].iloc[-1], "%Y-%m-%d")

                date_range = pd.date_range(
                    start=update_from, 
                    end=datetime.utcnow()
                )

                index = initial_data.index[-1]

                for date in tqdm(date_range):
                    
                    download = get_api_response(date=date)
                    new_data = extract_results(response=download, date = date, index = index)

                    dataframe = pd.concat(
                        [initial_data, new_data]
                    )
                    index += 1

                dataframe = dataframe.reset_index(drop=True)
                dataframe.to_parquet(path=DAILY_DATA_DIR/f"{base_currency}{target_currency}_{initial_start_date}_{datetime.utcnow()}.parquet")
                
                return dataframe 
            
            except:
                
                logger.error("Unable to update the file. Most likely, there is a problem with your Polygon subscription")
                
                return initial_data
            
        else:
            
            logger.info("There is no pre-existing dataset -> Defaulting to fetch data from the beginning of 2017 till date")
            
            # Download the data with the default values 
            dataframe = get_daily_ohlc()

            return dataframe


if __name__=="__main__":
    
    fire.Fire(get_daily_ohlc)
    