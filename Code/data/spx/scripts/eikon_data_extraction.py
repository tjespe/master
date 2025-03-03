# %% Pip instllation
#!pip install eikon
#!pip install more_itertools

# %% Importing libraries
import eikon as ek
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import time
from dateutil.relativedelta import relativedelta


# %% 1) set Eikon API key
ek.set_app_key('55d2b9814cee4ffdbc517b89b372b0a5436f3492')
ek.set_port_number(36036)

# %% 2) Define SPX sumbol
spx_symbol = ".SPXTR"


# %% 3) Define start and end
start_date = datetime(1990, 1, 1)
end_date = datetime(2025, 2, 19)

chunk_size = relativedelta(months=3) # We'll fetch data in  chunks to avoid hitting row limits
all_chunks = [] # List to accumulate each chunk's DataFrame




# %% 4) Loop over each date chunk
current_start = start_date
while current_start < end_date:
    chunk_end = current_start + chunk_size
    # Don't exceed the final end_date
    if chunk_end > end_date:
        chunk_end = end_date

    print(f"Fetching data from {current_start.date()} to {chunk_end.date()}...")
    # df_chunk = ek.get_timeseries(
    #     rics=[spx_symbol],
    #     fields=['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'], 
    #     start_date=current_start.strftime('%Y-%m-%d'),
    #     end_date=chunk_end.strftime('%Y-%m-%d'),
    #     interval='daily',
    #     corax='adjusted',
    # )

    # Retry mechanism
    for attempt in range(2):  # Try up to 2 times
        try:
            df_chunk = ek.get_timeseries( rics=[spx_symbol],
            fields=['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'], 
            start_date=current_start.strftime('%Y-%m-%d'),
            end_date=chunk_end.strftime('%Y-%m-%d'),
            interval='daily',
            corax='adjusted',)
            
            if df_chunk is not None and not df_chunk.empty:
                all_chunks.append(df_chunk)
                print(f"    ✅ Data retrieved: {df_chunk.shape[0]} rows")
            else:
                print(f"    ⚠️ No data returned for this period.")

            break  # Exit retry loop if successful

        except Exception as e:
            print(f"    ❌ Attempt {attempt + 1}: Error fetching data: {e}")
            if attempt < 1:
                time.sleep(10)  # Wait before retrying
    else:
        print(f"    ❌ Skipping {current_start.date()} to {chunk_end.date()} due to repeated errors.")


    # Advance to next chunk
    current_start = chunk_end

# %% 5) Combine all chunks into one wide DataFrame
if all_chunks:
    df_spx = pd.concat(all_chunks)
    # Drop any duplicated rows if an exact date boundary overlaps
    df_spx = df_spx[~df_spx.index.duplicated(keep='first')]
    
    df_spx.name = 'Date'

    # Flatten MultiIndex columns if they exist
    if isinstance(df_spx, pd.MultiIndex):
        df_spx.columns = ['_'.join(col).strip() for col in df_spx.columns.values]

    # Save or inspect the final DataFrame
    df_spx.to_csv("data/spx/raw_data/spx_19900101_to_today_20250219.csv")
    print("Data saved to data/spx/raw_data/spx_19900101_to_today_20250219.csv")
    print(df_spx.head())

else:
    print("No data was retrieved.")



# %%
