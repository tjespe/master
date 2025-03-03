# %% 1) Import packages
import time
import numpy as np
import pandas as pd

# %% 2) Read in the data and load the data
df_spx_data = pd.read_csv("data/spx/raw_data/spx_19900101_to_today_20250219.csv")
print("✅ Data loaded successfully.")
print("SPX data shape: ", df_spx_data.shape)

# %% 2) Dropp all rows where the date is before 1990-01-01 and after 2025-02-19
df_spx_data = df_spx_data[(df_spx_data["Date"] >= "1990-01-01") & (df_spx_data["Date"] <= "2025-02-19")]
print("✅ Rows before 1990-01-01 and after 2025-02-19 dropped successfully.")
print("SPX data shape: ", df_spx_data.shape)

# %% Drop all dubplicated rows in each data frame
df_spx_data = df_spx_data.drop_duplicates()
print("✅ Duplicated rows dropped successfully.")

# print how many rows where dropepd in each data frame
print("SPX data shape: ", df_spx_data.shape)


# %%  3) Display the first few rows of each DataFrame
print("SPX Data:")
df_spx_data

# %% show numbe of nan values in each colums
print("Number of NaN values in each column:")
print(df_spx_data.isnull().sum())


# %% 
df_spx_data[df_spx_data["CLOSE"].isnull()]

# %% Drop the rows with Nan valus for CLOSE
df_spx_data = df_spx_data.dropna(subset=["CLOSE"])
print("✅ Rows with NaN values in the CLOSE column dropped successfully.")

# print how many rows where dropepd in each data frame
print("SPX data shape: ", df_spx_data.shape)

# %% 
df_spx_data

# %% rename columns to be called Date, Close, High, Low, Open, Volume
df_spx_data.rename(columns={"CLOSE": "Close", "HIGH": "High", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"}, inplace=True)
df_spx_data

# %%
# Sort the dataframe by  Date in acending order
df_spx_data = df_spx_data.sort_values(by=["Date"], ascending=True)



# %% Double check the data by checking what dates are diffrent in dow jones data and spx.csv where where the date is before 1990-01-01 and after 2025-02-19
df_spx_data = pd.read_csv("data/spx/processed_data/spx_19900101_to_today_20250219_cleaned.csv")
df_dow_jones_data = pd.read_csv("data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

print("Dates in dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv but not in spx_19900101_to_today_20250219_cleaned.csv:")
print(df_dow_jones_data[~df_dow_jones_data["Date"].isin(df_spx_data["Date"])])
print("Dates in spx_19900101_to_today_20250219_cleaned.csv but not in dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv:")
print(df_spx_data[~df_spx_data["Date"].isin(df_dow_jones_data["Date"])])


# %%  Dropp all the endtires in the spx data that are not in the dow jones data
df_spx_data = df_spx_data[df_spx_data["Date"].isin(df_dow_jones_data["Date"])]
print("✅ Rows not in the dow jones data dropped successfully.")
print("SPX data shape: ", df_spx_data.shape)

df_spx_data

# double check that the dates are the same in both data frames
print("Dates in dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv but not in spx_19900101_to_today_20250219_cleaned.csv:")
print(df_dow_jones_data[~df_dow_jones_data["Date"].isin(df_spx_data["Date"])])
print("Dates in spx_19900101_to_today_20250219_cleaned.csv but not in dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv:")
print(df_spx_data[~df_spx_data["Date"].isin(df_dow_jones_data["Date"])])


# %% Save the final DataFrame to a CSV file
df_spx_data.to_csv("data/spx/processed_data/spx_19900101_to_today_20250219_cleaned.csv", index=False)

print("Data saved to data/spx/processed_data/spx_19900101_to_today_20250219_cleaned.csv")
print(df_spx_data.head())

# %%


