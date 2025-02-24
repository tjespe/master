# %%
import yfinance as yf
import pandas as pd
import numpy as np


dow_jones_tickers = [
    "AMZN",  # Amazon
    "AXP",  # American Express
    "AMGN",  # Amgen Inc.
    "AAPL",  # Apple Inc.
    "BA",  # Boeing
    "CAT",  # Caterpillar
    "CSCO",  # Cisco Systems
    "CVX",  # Chevron
    "GS",  # Goldman Sachs
    "HD",  # Home Depot
    "HON",  # Honeywell
    "IBM",  # International Business Machines (IBM)
    "JNJ",  # Johnson & Johnson
    "KO",  # Coca-Cola
    "JPM",  # JPMorgan Chase
    "MCD",  # McDonald's
    "MMM",  # 3M Company
    "MRK",  # Merck & Co.
    "MSFT",  # Microsoft
    "NKE",  # Nike
    "PG",  # Procter & Gamble
    "SHW",  # Sherwin-Williams (added in 2024, replacing Dow Inc.)
    "TRV",  # Travelers Companies
    "UNH",  # UnitedHealth Group
    "CRM",  # Salesforce
    "NVDA",  # Nvidia (added in 2024, replacing Intel)
    "VZ",  # Verizon Communications
    "V",  # Visa
    "WMT",  # Walmart
    "DIS"  # Walt Disney
]

print(len(dow_jones_tickers))

# %%
# make a general function to download data from yahoo finance
def get_data_yahoo(tickers, start, end) -> pd.DataFrame:
    # create an empty dataframe
    final_df = pd.DataFrame()
    # loop through all the tickers
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        data["Symbol"] = ticker
        data = data[["Symbol", "Close", "High", "Low", "Open", "Volume"]].reset_index()
        # set date and symbol as index
        data = data.set_index("Date")
        data.columns = data.columns.droplevel(1)
        data.columns = ["Symbol", "Close", "High", "Low", "Open", "Volume"]
        # rename close to "Adjusted Close"
        data = data.rename(columns={"Close": "Adjusted Close"})
        # calculate the daily returns and add it to the dataframe
        # data["Daily Return"] = data["Adjusted Close"].pct_change()
        # # calculate the daily log returns and add it to the dataframe
        final_df = pd.concat([final_df, data]) 
        print(f"Downloaded {ticker}")
    return final_df
# %%
# download all the data from the dow jones tickers
dow_jones = get_data_yahoo(dow_jones_tickers, "1990-01-01", "2025-02-19")
# set date and symbol as index
dow_jones = dow_jones.set_index(["Symbol"], append=True)
dow_jones
# %%
# save to csv
dow_jones.to_csv("../data/dow_jones_yahoo.csv")
# %%
