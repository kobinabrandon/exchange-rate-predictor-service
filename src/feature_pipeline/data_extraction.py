import os 
import requests      
   
import pandas as pd   
from tqdm import tqdm         

from datetime import datetime, timedelta

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
    index: int,
    base_currency: str = "GBP",
    target_currency: str = "GHS"
    ) -> pd.DataFrame:
    
    """ 
    We search through the value associated with the "results"
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
                        f"Opening_rate_{base_currency}{target_currency}": opening_rate,
                        f"Peak_rate_{base_currency}{target_currency}": peak_rate,
                        f"Lowest_rate_{base_currency}{target_currency}": lowest_rate,
                        f"Closing_rate_{base_currency}{target_currency}": closing_rate
                    }, index = [index]
                )
    

def get_daily_ohlc(
    start_date: datetime = datetime(2017,1,1), 
    end_date: datetime = datetime.today(),
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
    
    start_date_str = start_date.strftime(format="%Y-%m-%d")
    end_date_str = end_date.strftime(format="%Y-%m-%d")
    
    file_path = DAILY_DATA_DIR/f"{base_currency}{target_currency}_{start_date_str}_{end_date_str}.parquet"

    if file_path.exists():

        logger.info("The desired file already exists")

        dataframe = pd.read_parquet(file_path)

        return dataframe

    else:

        index = 0
        dataframe = pd.DataFrame()
        
        today = {
            "year": datetime.today().year,
            "month": datetime.today().month,
            "day": datetime.today().day
        }
        
        is_today = end_date.day == today["day"] and end_date.month == today["month"] and end_date.year == today["year"]
            
        # I could have used "if end_date == datetime.today()", however it doesn't work because datetime.today()
        # measures time down to the microsecond (I checked). Consequently, this if statement will always return false. 
        # I suspect that this happens because by the time the program gets to this if statement, the value of 
        # datetime.today() will have changed from what it was when the program was first executed.
        if is_today:
        
            if datetime.utcnow().hour < 16:

                date_range = pd.date_range(
                    start=start_date,
                    end=end_date - timedelta(days=1)
                )
            
            else:
                
                date_range = pd.date_range(
                    start=start_date,
                    end=end_date
                )
        
        else:
            
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



def update_ohlc(
    base_currency: str = "GBP",
    target_currency: str = "GHS"
    ) -> pd.DataFrame:
    
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
    
    logger.info("Looking for pre-existing files")
    
    with os.scandir(DAILY_DATA_DIR) as data:
        
        if any(data):
            
            logger.info("Found a file -> Getting the most recent of these files")

            initial_data = get_newest_local_dataset()

            logger.info("Checking whether the file is up-to-date")

            today = datetime.today().strftime("%Y-%m-%d")

            # This is the date from which the file will be updated if necessary
            update_from = initial_data["Date"].iloc[-1]

            # Check whether the last date in the file corresponds to today
            if update_from == today:

                logger.info("The file is up-to-date")

                return initial_data

            else:

                logger.info(f"The file was up-to-date as of {update_from} -> Updating it")

                def UpdateFailed(Exception):

                    pass

                try:

                    date_range = pd.date_range(
                        start=update_from,
                        end=today
                    )

                    index = initial_data.index[-1]

                    for date in tqdm(date_range):

                        download = get_api_response(date=date)

                        new_data = extract_results(
                            response=download,
                            date = date,
                            index = index
                            )

                        dataframe = pd.concat(
                            objs=[initial_data, new_data]
                        )

                        index += 1

                    initial_start_date = initial_data["Date"].iloc[0]

                    dataframe = dataframe.reset_index(drop=True)
                    dataframe.to_parquet(path=DAILY_DATA_DIR/f"{base_currency}{target_currency}_{initial_start_date}_{today}.parquet")

                    return dataframe

                except UpdateFailed:

                    return initial_data
            
        else:
            
            logger.info("No dataset has been saved -> Defaulting to fetch data from the beginning of 2017 till date")
            
            # Download the data with the default values 
            dataframe = get_daily_ohlc()

            return dataframe
