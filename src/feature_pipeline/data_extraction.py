import os
import glob
import requests

import pandas as pd 
from tqdm import tqdm
from loguru import logger
from datetime import datetime, timedelta

from src.config import config
from src.paths import HISTORICAL_DATA


def get_raw_data_for_one_day(year: int, month: int, day: int, base_currency: str) -> dict[str, str|int|float]: 
    """
    We fetch the exchange rates from the ExchangeRate-API.


    Args:
        year (int): _description_
        month (int): _description_
        day (int): _description_
        base_currency (str): _description_

    Returns:
        dict[str, str|int|float]: _description_
    """
    url = f"https://v6.exchangerate-api.com/v6/{config.exchange_rate_api_key}/history/{base_currency}/{year}/{month}/{day}"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Error {response.status_code} - {response.text}")
    else:
        return response.json()


def make_daily_dataframe(response: dict, base_currency: str) -> pd.DataFrame:
    """
    We search through the value associated with the "results" key in the API response, 
    parse it for the  OHLC data, and put it all into a dataframe.

    Returns:
        pd.DataFrame: all the OHLC data from the given start date to the end date.
    """
    if response["result"] == "success":
        data = pd.DataFrame(response)
        data = data.drop(["result", "documentation", "terms_of_use", "year", "month", "day", "base_code"], axis=1)
        transposed_data = data.T.drop(base_currency, axis=1)
        transposed_data["Date"] = f"{response["day"]}-{response["month"]}-{response["year"]}"
        return transposed_data
    else:
        raise Exception("There request to the API was unsuccessful")


def get_historical_data(start_date: datetime, end_date: datetime = datetime.today()) -> pd.DataFrame:
    """
    By default, download currency exchange rate data with pounds sterling and the Ghanaian cedi as the base and 
    target currencies respectively.
    
    Args:
        start_date: the date from which we want to collect data
        end_date: the last date on which we want data
    
    Returns:
        pd.DataFrame: a dataframe of daily exchange rates during the specified period               
    """
    start_date_str = start_date.strftime(format="%Y-%m-%d")
    end_date_str = end_date.strftime(format="%Y-%m-%d")
    file_path = HISTORICAL_DATA / f"{base_currency}{start_date_str}_{end_date_str}.parquet"

    if file_path.exists():
        logger.info(f"Historical data for {base_currency} already exists")
        dataframe = pd.read_parquet(file_path)
        return dataframe
    else:
        index = 0
        dataframe = pd.DataFrame()
        if is_today(date=end_date):

            # if it is currently, before 5PM, the program should refrain from downloading today's data.
            if is_closed():
                end_date = end_date - timedelta(days=1)

        date_range = pd.date_range(start=start_date, end=end_date)

        for date in tqdm(date_range):

            # The forex market closes on Saturdays (which the datetime package sees as day 5)
            if datetime.weekday(date) != 5:
                api_response = get_raw_data_for_one_day(date=date)
                current_date_data = make_daily_dataframe(response=api_response, date=date, index=index)
                dataframe = pd.concat([dataframe, current_date_data], axis=0)
                index += 1

        dataframe = dataframe.reset_index(drop=True)
        dataframe.to_parquet(path=file_path)
        return dataframe


    
def is_today(date: datetime) -> bool:
    """
    I could have said "if date == datetime.today()", however it doesn't work because datetime.today()
    measures time down to the microsecond (I checked). Consequently, the returned boolean will always 
    be True.  I suspect that this happens because by the time the program gets to this if statement, 
    the value of datetime.today() will have changed from what it was when the program was first executed.

    Args:
        date: the date to be checked.
    
    Returns:
        bool: returns True if the entered date is today, and False if it is not.
    """
    today = {
        "year": datetime.today().year,
        "month": datetime.today().month,
        "day": datetime.today().day
    }

    return date.day == today["day"] and date.month == today["month"] and date.year == today["year"]


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
    with os.scandir(f"{HISTORICAL_DATA}") as data:

        if any(data):
            files = glob.glob(f"{HISTORICAL_DATA}/*.parquet")
            newest_file = max(files, key=os.path.getctime)
            dataframe = pd.read_parquet(newest_file)

        else:
            logger.info("There is no file saved in local storage -> Fetching data from 2017 till date.")
            dataframe = get_historical_data()

        return dataframe


def update_ohlc(base_currency: str = "GBP", target_currency: str = "GHS") -> pd.DataFrame:
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

    if any(os.scandir(HISTORICAL_DATA)):

        logger.info("Getting the most recent file in local storage")
        initial_data = get_newest_local_dataset()
        logger.info("Checking whether the file is up-to-date")
        today = datetime.today()

        # This is the date from which the file will be updated if necessary
        update_from = datetime.strptime(date_string=initial_data["Date"].iloc[-1], format="%Y-%m-%d")

        # Check whether the last date in the file corresponds to today
        if is_today(date=update_from):
            logger.info("The file is up-to-date")
            return initial_data

        else:
            logger.info(f"The file was up-to-date as of {update_from} -> Updating it")

            to_download = pd.date_range(start=update_from + timedelta(days=1), end=today)
            index = initial_data.index[-1]
            dataframe = pd.DataFrame()

            for date in tqdm(to_download):
                if is_today(date=date) and is_closed():
                    logger.info("There is nothing to download")
                    break 

                download = get_raw_data_for_one_day(date=date)
                new_data = make_daily_dataframe(response=download, date=date, index=index)

                if new_data is None:
                    continue

                dataframe = pd.concat(objs=[dataframe, new_data], axis=0)
                index += 1

            updated_data = pd.concat(objs=[initial_data, dataframe], axis=0)
            updated_data = updated_data.reset_index(drop=True)

            initial_start_date = initial_data["Date"].iloc[0]
            today_str = today.strftime(format="%Y-%m-%d")

            updated_data.to_parquet(
                path=HISTORICAL_DATA / f"{base_currency}{target_currency}_{initial_start_date}_{today_str}.parquet"
            )

            return updated_data

    else:

        logger.info("No dataset has been saved -> Fetching data from the beginning of 2017 till date by default")

        # Download the data with the default values 
        dataframe = get_historical_data()

        return dataframe


result = get_raw_data_for_one_day(year=2024, month=9, day=20, base_currency="GHS")
data = make_daily_dataframe(response=result, base_currency="GHS")
breakpoint()