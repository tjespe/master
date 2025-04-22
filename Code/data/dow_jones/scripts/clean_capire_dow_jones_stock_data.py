# %% 1) Import packages
import time
import numpy as np
import pandas as pd

# %% 2) Load Excel file
file_path = "data/dow_jones/raw_data/Capire_March2024_dow_jones.xlsx"
xls = pd.ExcelFile(file_path)
print("✅ Excel file loaded successfully.")
print("Sheet names:", xls.sheet_names)

# %% 3) Load Companies and Dates sheets
df_companies = pd.read_excel(xls, sheet_name="Companies", header=None)
df_dates = pd.read_excel(xls, sheet_name="Dates", header=None)

# Convert dates to YYYY-MM-DD format
df_dates[0] = pd.to_datetime(df_dates[0], format="%d-%b-%Y").dt.strftime("%Y-%m-%d")

# Extract lists of companies and dates
companies = df_companies[0].tolist()
dates = df_dates[0].tolist()

print("✅ Companies and Dates extracted successfully.")
print("Total Companies:", len(companies))
print("Total Dates:", len(dates))

# %% 4) Load data sheets
data_sheets = ["RV", "BPV", "Good", "Bad", "RQ", "RV_5", "BPV_5", "Good_5", "Bad_5", "RQ_5"]
data_dict = {}

for sheet in data_sheets:
    data_dict[sheet] = pd.read_excel(xls, sheet_name=sheet, header=None)
    print(f"✅ {sheet} data loaded successfully. Shape: {data_dict[sheet].shape}")

# %% 5) Reshape data into long format
final_data = []

for i, date in enumerate(dates):
    for j, company in enumerate(companies):
        row = [date, company] + [data_dict[sheet].iloc[i, j] for sheet in data_sheets]
        final_data.append(row)

# Define column names
columns = ["Date", "Symbol"] + data_sheets

# Create DataFrame
df_final = pd.DataFrame(final_data, columns=columns)
print("✅ Data reshaped successfully.")
print("Final Data Shape:", df_final.shape)

# %% 6) Save the cleaned data to CSV
output_file = "data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv"
df_final.to_csv(output_file, index=False)
print(f"✅ Processed data saved successfully: {output_file}")

# %%
