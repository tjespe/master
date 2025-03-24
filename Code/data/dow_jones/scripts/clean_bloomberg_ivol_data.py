# %% 1) Import packages
import time
import numpy as np
import pandas as pd

# %% 2) Load Excel file
# file_path = "Code/data/dow_jones/raw_data/Bloomberg_IVOL_03012025_to_11032025.xlsx"
file_path = "../raw_data/Bloomberg_IVOL_03012025_to_11032025.xlsx"
xls = pd.ExcelFile(file_path)
print("✅ Excel file loaded successfully.")
print("Sheet names:", xls.sheet_names)
# %%

final_data = []

# Process each sheet
for sheet in xls.sheet_names:
    print(f"Processing sheet: {sheet}...")
    df = pd.read_excel(xls, sheet_name=sheet, header=None)

    # The tickers are in the second row (index 1), one for every two columns
    tickers = df.iloc[1, ::2].tolist()

    # Remove the first two rows (headers)
    df = df.iloc[2:, :]

    # Iterate over ticker/date-value column pairs
    for idx, ticker in enumerate(tickers):
        date_col = idx * 2
        value_col = date_col + 1

        for i in range(df.shape[0]):
            date = df.iloc[i, date_col]
            value = df.iloc[i, value_col]
            final_data.append([date, ticker, sheet, value])

# Convert to DataFrame
columns = ["Date", "Symbol", "Sheet", "Value"]
df_temp = pd.DataFrame(final_data, columns=columns)
# %%
df_temp

# %%
# Convert Date to proper format
df_temp["Date"] = pd.to_datetime(df_temp["Date"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")

#%%
df_temp

# %%
# Pivot the DataFrame
df_wide = df_temp.pivot_table(
    index=["Date", "Symbol"],
    columns="Sheet",
    values="Value",
    aggfunc="first"  # in case of duplicates
).reset_index()

# Optional: flatten the MultiIndex columns if needed
df_wide.columns.name = None  # remove the "Sheet" name from columns
df_wide.columns = [str(col) for col in df_wide.columns]

df_wide

# %% 
# Sort by symbol and date
df_wide = df_wide.sort_values(by=["Symbol", "Date"]).reset_index(drop=True)
df_wide


# %%
# Save the processed data to CSV
output_file = "../processed_data/processed_ivol_data.csv"
df_wide.to_csv(output_file, index=False)
print(f"✅ Processed data saved successfully: {output_file}")

# %%

# %%
