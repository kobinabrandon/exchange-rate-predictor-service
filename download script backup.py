

def update_rates(data: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    
    new_data = pd.DataFrame()
    date_range = pd.date_range(start=start_date, end=end_date)
    index = 0
    
    for date in tqdm(date_range):
        
        URL = f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/fx/{date.strftime('%Y-%m-%d')}?adjusted=true&apiKey={POLYGON_API_KEY}"

        endpoint = requests.get(URL)

        if endpoint.status_code != 200:
            
            print(f"Error {endpoint.status_code} - {endpoint.text}")
            
        else:

            download = endpoint.json()  

            if "results" in download.keys():

                for results in download["results"]:

                    if results["T"] == f"C:GBPGHS":
                        
                        opening_rate = results["o"]
                        peak_rate = results["h"]
                        lowest_rate = results["l"]
                        closing_rate = results["c"]
                        
                        current_date_data = pd.DataFrame(
                            {
                                "Date": date.strftime("%Y-%m-%d"),
                                f"Opening rate (GBPGHS)": opening_rate,
                                f"Peak rate (GBPGHS)": peak_rate,
                                f"Lowest rate (GBPGHS)": lowest_rate,
                                f"Closing rate (GBPGHS)": closing_rate
                            }, index = [index]
                        )

                        dataframe = pd.concat([dataframe, new_data])
                        index += 1

        dataframe.to_parquet(path=DAILY_DATA_DIR/f"GBPGHS_{start_date}_{end_date}.parquet")

        return dataframe
