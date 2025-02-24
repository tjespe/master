# %% Pip installation
#!pip install refinitiv-data

# %% Importing libraries
import refinitiv.data as rd
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import time
from dateutil.relativedelta import relativedelta

# %% 1) Open Refinitiv Data API session
rd.open_session()

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



# %% 3) Define start and end dates
start_date = datetime(1990, 1, 1)
end_date = datetime.now()
chunk_size = relativedelta(months=6)  # Larger chunk size to reduce requests

# %% 4) Storage for all data
all_data = []

# %% 5) Outer loop: Process one ticker at a time
for symbol in dow_jones_symbols:
    print(f"\nFetching data for {symbol}...")

    current_start = start_date
    symbol_data = []

    # Inner loop: Fetch data in chunks
    while current_start < end_date:
        chunk_end = current_start + chunk_size
        if chunk_end > end_date:
            chunk_end = end_date

        print(f"  - Fetching from {current_start.date()} to {chunk_end.date()}...")

        # API request parameters
        kwargs = {
            "universe": symbol,
            "fields": ["TR.TotalReturn"],
            "start": current_start.strftime('%Y-%m-%d'),
            "end": chunk_end.strftime('%Y-%m-%d'),
            "interval": "daily"
        }

        # Retry mechanism: Try fetching twice before skipping
        for attempt in range(2):  # Try up to 2 times
            try:
                df_chunk = rd.get_history(**kwargs)
                
                if df_chunk is not None and not df_chunk.empty:
                    df_chunk = df_chunk.reset_index()  # Ensure 'Date' is a column
                    df_chunk["Symbol"] = symbol  # Add symbol column
                    symbol_data.append(df_chunk)
                    print(f"    ✅ Data retrieved: {df_chunk.shape[0]} rows")
                else:
                    print(f"    ⚠️ No data returned for {symbol} in this period.")

                break  # Exit retry loop if successful

            except Exception as e:
                print(f"    ❌ Attempt {attempt + 1}: Error fetching data: {e}")
                if attempt < 1:  # Only sleep before retry, not after last attempt
                    time.sleep(10)  # Wait before retrying

        else:  # If both attempts fail, skip this period
            print(f"    ❌ Skipping {symbol} from {current_start.date()} to {chunk_end.date()} due to repeated errors.")

        # Move to the next chunk
        current_start = chunk_end

    # Combine all chunks for this symbol
    if symbol_data:
        df_symbol = pd.concat(symbol_data)
        all_data.append(df_symbol)

# %% 6) Combine all symbols into one DataFrame and save
if all_data:
    df_final = pd.concat(all_data)
    df_final.index.name = 'Date'

    # Ensure 'Date' is included
    if 'Date' not in df_final.columns:
        df_final = df_final.reset_index()

    # Flatten MultiIndex columns if they exist
    if isinstance(df_final.columns, pd.MultiIndex):
        df_final.columns = ['_'.join(col).strip() for col in df_final.columns.values]

    # Save to CSV
    output_path = "data/dow_jones/raw_data/dow_jones_stocks_total_return_1990_to_today_19022025.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print("\n✅ Data saved successfully:", output_path)
    print(df_final.head())
else:
    print("\n⚠️ No data retrieved.")


