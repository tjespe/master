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

# %% 2) Dow Jones constituents (RICs)
dow_jones_symbols = [
    "AAPL.O",   # Apple (NASDAQ)
    "AMGN.O",   # Amgen (NASDAQ)
    "AMZN.O",   # Amazon (NASDAQ)
    "AXP",      # American Express (NYSE)
    "BA",       # Boeing (NYSE)
    "CAT",      # Caterpillar (NYSE)
    "CRM",      # Salesforce (NYSE)
    "CSCO.O",   # Cisco (NASDAQ)
    "CVX",      # Chevron (NYSE)
    "DIS",      # Disney (NYSE)
    "DOW",      # Dow Inc. (NYSE)
    "GS",       # Goldman Sachs (NYSE)
    "HD",       # Home Depot (NYSE)
    "HON.O",    # Honeywell (NASDAQ)
    "IBM",      # IBM (NYSE)
    "INTC.O",   # Intel (NASDAQ)
    "JNJ",      # Johnson & Johnson (NYSE)
    "JPM",      # JPMorgan Chase (NYSE)
    "KO",       # Coca-Cola (NYSE)
    "MCD",      # McDonald’s (NYSE)
    "MMM",      # 3M (NYSE)
    "MRK",      # Merck & Co. (NYSE)
    "MSFT.O",   # Microsoft (NASDAQ)
    "NKE",      # Nike (NYSE)
    "PG",       # Procter & Gamble (NYSE)
    "TRV",      # Travelers (NYSE)
    "UNH",      # UnitedHealth (NYSE)
    "V",        # Visa (NYSE)
    "VZ",       # Verizon (NYSE)
    "WMT"       # Walmart (NYSE)
]
# %% 3) Define start and end
start_date = datetime(1990, 1, 1)
end_date = datetime.now()

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
    #     rics=dow_jones_symbols,
    #     fields=['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'], 
    #     start_date=current_start.strftime('%Y-%m-%d'),
    #     end_date=chunk_end.strftime('%Y-%m-%d'),
    #     interval='daily',
    #     corax='adjusted',
    # )

    # Retry mechanism
    for attempt in range(2):  # Try up to 2 times
        try:
            df_chunk = ek.get_timeseries( rics=dow_jones_symbols,
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
    df_wide = pd.concat(all_chunks)
    # Drop any duplicated rows if an exact date boundary overlaps
    df_wide = df_wide[~df_wide.index.duplicated(keep='first')]
    
    df_wide.index.name = 'Date'

    # Flatten MultiIndex columns if they exist
    if isinstance(df_wide.columns, pd.MultiIndex):
        df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]


    # Melt the DataFrame into long format
    df_long = df_wide.reset_index().melt(
        id_vars=['Date'],
        var_name='Symbol_Metric',
        value_name='Value'
    )

    # Extract Symbol and Metric separately
    df_long[['Symbol', 'Metric']] = df_long['Symbol_Metric'].str.rsplit('_', n=1, expand=True)

    # Pivot the table to arrange columns in the required format
    df_final = df_long.pivot_table(index=['Date', 'Symbol'], columns='Metric', values='Value', aggfunc='first').reset_index()

    # Ensure correct column order (including ADJUST_CLS)
    desired_order = ['Date', 'Symbol', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
    df_final = df_final.reindex(columns=[col for col in desired_order if col in df_final.columns])


    # 7) Save or inspect the final DataFrame
    df_final.to_csv("data/dow_jones/raw_data/dow_jones_stocks_19022025v2.csv", index=False)
    print("Data saved to data/dow_jones/raw_data/dow_jones_stocks_19022025v2.csv")
    print(df_final.head())

else:
    print("No data was retrieved.")



# %%
