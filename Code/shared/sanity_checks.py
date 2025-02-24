# %%
import pandas as pd
import numpy as np

# %%
# load the following data Code\data\dow_jones\processed_data\dow_jones_stocks_1990_to_today_19022025_cleaned.csv
dow_jones_eikon = pd.read_csv("../data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned.csv")
dow_jones_yahoo = pd.read_csv("../data/dow_jones_yahoo.csv")

# index both dataframes on the date and symbol
dow_jones_eikon = dow_jones_eikon.set_index(["Date", "Symbol"])
dow_jones_yahoo = dow_jones_yahoo.set_index(["Date", "Symbol"])

# %%
dow_jones_eikon

# %%
dow_jones_yahoo

# %%
# Get data for apple from both dataframes
apple_eikon = dow_jones_eikon.loc[(slice(None), "AAPL.O"), :]
apple_yahoo = dow_jones_yahoo.loc[(slice(None), "AAPL"), :]

# count the number of entries in both dataframes
print("Lenght apple eikon", len(apple_eikon))
print("Lenght apple yahoo", len(apple_yahoo))

# remove symbol coloumns and total Return Coloumn
apple_eikon = apple_eikon.drop(columns=["Total Return"])
# remove symbol as index and drop it
apple_eikon = apple_eikon.reset_index().drop(columns=["Symbol"])
apple_yahoo = apple_yahoo.reset_index().drop(columns=["Symbol"])

#rename all coloumns so that is says _e for eikon and _y for yahoo except for the Date coloumn
apple_eikon.columns = [col + "_e" if col != "Date" else col for col in apple_eikon.columns]
apple_yahoo.columns = [col + "_y" if col != "Date" else col for col in apple_yahoo.columns]

print("Apple eikon")
apple_eikon
# %%
print("Apple yahoo")
apple_yahoo

# %%
# print the number of NaN dates in both dataframes
print("Number of NaN dates in apple eikon", apple_eikon["Date"].isna().sum())
print("Number of NaN dates in apple yahoo", apple_yahoo["Date"].isna().sum())

# find out which dates are in the yahoo dataframe but not in the eikon dataframe
dates_not_in_eikon = apple_yahoo[~apple_yahoo["Date"].isin(apple_eikon["Date"])]
print("Dates not in eikon")
dates_not_in_eikon
# %%
# merge the two dataframes on the date
apple_merged = apple_eikon.merge(apple_yahoo, on="Date", how="inner")
apple_merged
# %%
# make the Volume_e coloumn not shoe scientific notation, only that coloumn
pd.options.display.float_format = '{:.2f}'.format
apple_merged

# %%
# make this the order of coloumns: Date, Close_e, Adjusted Close_y, High_e, High_y, Low_e, Low_y, Open_e, Open_y, Volume_e, Volume_y
apple_merged = apple_merged[["Date", "Close_e", "Adjusted Close_y", "High_e", "High_y", "Low_e", "Low_y", "Open_e", "Open_y", "Volume_e", "Volume_y"]]
# %%
# print the dataframe and make it scrollable
from IPython.display import display
from IPython.display import HTML

display(HTML(apple_merged.to_html(index=False)))
# %%
# Let's do all the same for "WMT"
walmart_eikon = dow_jones_eikon.loc[(slice(None), "WMT"), :]
walmart_yahoo = dow_jones_yahoo.loc[(slice(None), "WMT"), :]
walmart_eikon = walmart_eikon.drop(columns=["Total Return"])
walmart_eikon = walmart_eikon.reset_index().drop(columns=["Symbol"])
walmart_yahoo = walmart_yahoo.reset_index().drop(columns=["Symbol"])
walmart_eikon.columns = [col + "_e" if col != "Date" else col for col in walmart_eikon.columns]
walmart_yahoo.columns = [col + "_y" if col != "Date" else col for col in walmart_yahoo.columns]
print("Walmart eikon")
walmart_eikon

# %%
print("Walmart yahoo")
walmart_yahoo

# %%
print("Number of NaN dates in walmart eikon", walmart_eikon["Date"].isna().sum())
print("Number of NaN dates in walmart yahoo", walmart_yahoo["Date"].isna().sum())
dates_not_in_eikon = walmart_yahoo[~walmart_yahoo["Date"].isin(walmart_eikon["Date"])]
print("Dates not in eikon")
dates_not_in_eikon

# %%
walmart_merged = walmart_eikon.merge(walmart_yahoo, on="Date", how="inner")
walmart_merged


# %%
# make a dictionary with the keys and the number of nan values in the eikon dataframe for that ticker as the value
nan_values = {}
for ticker in dow_jones_eikon.index.get_level_values("Symbol").unique():
    data = dow_jones_eikon.loc[(slice(None), ticker), :]
    nan_values[ticker] = data.index.get_level_values("Date").isna().sum()

# print the dictionary
nan_values
# %%
# print the total number of NaN values in the eikon dataframe
sum(nan_values.values())
# %%
# make a dictionary with the keys and the number of nan values in the yahoo dataframe for that ticker as the value
nan_values = {}
for ticker in dow_jones_yahoo.index.get_level_values("Symbol").unique():
    data = dow_jones_eikon.loc[(slice(None), ticker), :]
    nan_values[ticker] = data["Date"].isna().sum()

# print the dictionary
nan_values
# %%
# print the total number of NaN values in the yahoo dataframe
sum(nan_values.values())
# %%
