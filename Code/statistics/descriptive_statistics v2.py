
#%% Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from shared.processing import get_lstm_train_test_new


#%% 
# # Load raw data
df_return_eikon = pd.read_csv("data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned.csv")
df_rv_capire = pd.read_csv("data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
df_iv_bloomberg = pd.read_csv("data/dow_jones/processed_data/processed_ivol_data.csv")


# %%
df_return_eikon
# %%
df_rv_capire
# %%
df_iv_bloomberg



# %%
# Remove all data in df_return_eikon tht is from before the first data and after the last date in df_rv_capire
df_return_eikon = df_return_eikon[(df_return_eikon['Date'] >= df_rv_capire['Date'].min()) & (df_return_eikon['Date'] <= df_rv_capire['Date'].max())]
# Remove all data in df_iv_bloomberg that is from before the first data and after the last date in df_rv_capire
df_iv_bloomberg = df_iv_bloomberg[(df_iv_bloomberg['Date'] >= df_rv_capire['Date'].min()) & (df_iv_bloomberg['Date'] <= df_rv_capire['Date'].max())]

# %% 
# Remove ".O" from the end of the Symbol in df_return_eikon
df_return_eikon['Symbol'] = df_return_eikon['Symbol'].str.replace('.O', '', regex=False)

# %%
# Display the firt and last date in each dataframe
print("First date in df_return_eikon: ", df_return_eikon['Date'].min())
print("Last date in df_return_eikon: ", df_return_eikon['Date'].max())
print("First date in df_rv_capire: ", df_rv_capire['Date'].min())
print("Last date in df_rv_capire: ", df_rv_capire['Date'].max())
print("First date in df_iv_bloomberg: ", df_iv_bloomberg['Date'].min())
print("Last date in df_iv_bloomberg: ", df_iv_bloomberg['Date'].max())

# %%
# Display count of values for each of the stocks for each varaible for alle thrree datasets
print("Count of values for each of the stocks for each varaible for df_return_eikon: ")
print(df_return_eikon.groupby('Symbol').count())
print("Count of values for each of the stocks for each varaible for df_rv_capire: ")
print(df_rv_capire.groupby('Symbol').count())
print("Count of values for each of the stocks for each varaible for df_iv_bloomberg: ")
print(df_iv_bloomberg.groupby('Symbol').count())



# %%
df_return_eikon
# %%
df_rv_capire
# %%
df_iv_bloomberg
# %%

# Left join df_rv_capire and df_return_eikon and then left join the result with df_iv_bloomberg on the Date and Symbol columns
df_combined = pd.merge(df_rv_capire, df_return_eikon, on=['Date', 'Symbol'], how='left')
df_combined = pd.merge(df_combined, df_iv_bloomberg, on=['Date', 'Symbol'], how='left')
# %%
df_combined
# %%




# Import necessary libraries
import pandas as pd
import numpy as np

# Define the descriptive stats function
def descriptive_stats(df):
    numeric_df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    stats = pd.DataFrame(index=numeric_df.columns)
    stats['Mean'] = numeric_df.mean()
    stats['Std'] = numeric_df.std()
    stats['Min'] = numeric_df.min()
    stats['Median'] = numeric_df.median()
    stats['Max'] = numeric_df.max()
    stats['Skewness'] = numeric_df.skew()
    stats['Kurtosis'] = numeric_df.kurtosis()
    return stats


# Group by Symbol and calculate stats per asset
grouped = df_combined.groupby('Symbol')
stats_per_asset = grouped.apply(lambda x: descriptive_stats(x))

# Average the stats across assets
average_stats = stats_per_asset.groupby(stats_per_asset.index.get_level_values(1)).mean()
sds = stats_per_asset.groupby(stats_per_asset.index.get_level_values(1)).std()

# Round for readability
average_stats = average_stats.round(3)
sds = sds.round(3)

# Format average ± std for all stats except Count
average_stats_formatted = average_stats.applymap(lambda x: f"{x:.2f}") + " {\\tiny$\\pm$ " + sds.applymap(lambda x: f"{x:.2f}") + "}"

# Add Count separately
# Count non-zero and non-NaN values per column (not grouped by asset)
# Count logic: include zeros for 'Total Return', exclude zeros for others
def custom_count(series):
    if series.name == "Total Return":
        return series.notna().sum()  # Include zeros
    else:
        return ((series != 0) & series.notna()).sum()  # Exclude zeros

counts = df_combined.select_dtypes(include=[np.number]).apply(custom_count)
counts = counts.astype(str)
average_stats_formatted.insert(0, "Count", counts[average_stats_formatted.index])


# Put Count first
cols_order = ["Count", "Mean", "Std", "Min", "Median", "Max", "Skewness", "Kurtosis"]
average_stats_formatted = average_stats_formatted[cols_order]

# # 1. Map raw names to pretty LaTeX names
latex_name_map = {
    "Total Return": "Return",
    "RV": "RV\\textsubscript{1-min}",
    "RV_5": "RV\\textsubscript{5-min}",
    "BPV": "BPV\\textsubscript{1-min}",
    "BPV_5": "BPV\\textsubscript{5-min}",
    "Good": "RV\\textsuperscript{+}\\textsubscript{1-min}",
    "Good_5": "RV\\textsuperscript{+}\\textsubscript{5-min}",
    "Bad": "RV\\textsuperscript{-}\\textsubscript{1-min}",
    "Bad_5": "RV\\textsuperscript{-}\\textsubscript{5-min}",
    "RQ": "RQ\\textsubscript{1-min}",
    "RQ_5": "RQ\\textsubscript{5-min}",
    "10 Day Call IVOL": "IV\\textsubscript{10-day}",
    "Historical Call IVOL": "IV\\textsubscript{20-day}",
}

# 2. Your strict manual order
desired_main_order = [
    "Total Return", 
    "RV", 
    "RV_5", 
    "BPV", 
    "BPV_5", 
    "Good", 
    "Good_5", 
    "Bad", 
    "Bad_5", 
    "RQ", 
    "RQ_5", 
    "10 Day Call IVOL", 
    "Historical Call IVOL"
]

# Rename the index using LaTeX names
average_stats_formatted.rename(index=latex_name_map, inplace=True)

# Apply final order
final_order = [latex_name_map[col] for col in desired_main_order if col in latex_name_map]
average_stats_formatted = average_stats_formatted.loc[final_order]


# Fill empty Skewness and Kurtosis for non-Return rows if needed (your original intention)
average_stats_formatted.loc[average_stats_formatted.index != "Return", ["Skewness", "Kurtosis"]] = ""
#%%
# Display
print("Table 2: Average Descriptive Statistics of Target Variable Across Assets\n")

# Display the formatted DataFrame, but displey only the Return row
average_stats_formatted_return = average_stats_formatted.loc[["Return"]]
print(average_stats_formatted_return.to_latex(index=True, escape=False))

#%%
# Display the formatted DataFrame, but display all rows exept the Return row and remove the skewness and kurtosis columns
average_stats_formatted_no_return = average_stats_formatted.drop(index=["Return"])
average_stats_formatted_no_return = average_stats_formatted_no_return.drop(columns=["Skewness", "Kurtosis"])
print(average_stats_formatted_no_return.to_latex(index=True, escape=False))


# %% Gemerate LaTeX code for descriptive statistics tables per stock

cols_to_keep = ["Count", "Mean", "Min", "Median", "Max", "Skewness", "Kurtosis"]

ticker_to_name = {
    "AAPL": "Apple Inc.",
    "AMGN": "Amgen Inc.",
    "AMZN": "Amazon.com Inc.",
    "AXP": "American Express Company",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "CRM": "Salesforce Inc.",
    "CSCO": "Cisco Systems Inc.",
    "CVX": "Chevron Corporation",
    "DIS": "The Walt Disney Company",
    "DOW": "Dow Inc.",
    "GS": "Goldman Sachs Group Inc.",
    "HD": "Home Depot Inc.",
    "HON": "Honeywell International Inc.",
    "IBM": "International Business Machines Corporation",
    "INTC": "Intel Corporation",
    "JNJ": "Johnson \& Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "KO": "The Coca-Cola Company",
    "MCD": "McDonald’s Corporation",
    "MMM": "3M Company",
    "MRK": "Merck \& Co. Inc.",
    "MSFT": "Microsoft Corporation",
    "NKE": "NIKE Inc.",
    "PG": "Procter \& Gamble Company",
    "TRV": "Travelers Companies Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "V": "Visa Inc.",
    "VZ": "Verizon Communications Inc.",
    "WMT": "Walmart Inc.",
}

latex_blocks = []

# Generate LaTeX code
for symbol, group_df in df_combined.groupby("Symbol"):
    stats = descriptive_stats(group_df)
    # Count of non-zero and non-NaN values - but if the series is Return, count all non-null values
    if "Total Return" in group_df.columns:
        counts = group_df["Total Return"].notna().sum()
    else:
        # For other series, count non-zero and non-NaN values
        counts = group_df.select_dtypes(include=[np.number]).apply(lambda col: ((col != 0) & (~col.isna())).sum())
    #counts = group_df.select_dtypes(include=[np.number]).apply(lambda col: ((col != 0) & (~col.isna())).sum())
    stats.insert(0, "Count", counts)
    stats = stats[cols_to_keep]
    stats = stats.round(2)
    stats = stats.applymap(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))

    stats.rename(index=latex_name_map, inplace=True)
    stats = stats.loc[[latex_name_map[col] for col in desired_main_order if col in latex_name_map]]
    stats["Skewness"] = stats["Skewness"].astype(object)
    stats["Kurtosis"] = stats["Kurtosis"].astype(object)
    stats.loc[stats.index != "Return", ["Skewness", "Kurtosis"]] = ""

    body = stats.to_latex(index=True, escape=False, header=True, column_format="lrrrrrll").splitlines()
    body = "\n".join(body[3:-2])  # Remove the \begin{tabular} and \end{tabular}

    company_name = ticker_to_name.get(symbol, symbol)
    caption = f"{company_name} ({symbol})"
    table_code = (
        "\\begin{minipage}{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\caption*{{{caption}}}\n"
        "    \\vspace{-0.4em}\n"
        "    \\scriptsize\n"
        "    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\renewcommand{\\arraystretch}{0.95}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{lrrrrrrr}\n"
        "        \\toprule\n"
        "        & Count & Mean & Min & Median & Max & Skew & Kurtosis \\\\\n"
        f"      {body}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}%\n"
        "    }\n"
        "\\end{minipage}%\n"
    )
    latex_blocks.append(table_code)

# Combine into figures with 2 tables per row, 5 rows per figure
grouped_figures = []

for i in range(0, len(latex_blocks), 10):  # 10 tables per figure (2 x 5 grid)
    chunk = latex_blocks[i:i+10]
    figure_lines = ["\\begin{figure}[H]", "\\centering"]

    for j in range(0, len(chunk), 2):
        left_table = chunk[j]
        right_table = chunk[j+1] if j+1 < len(chunk) else None

        # Append the left table
        figure_lines.append(left_table.strip())

        # If there's a right table, add hfill and then it
        if right_table:
            figure_lines.append("\\hfill")
            figure_lines.append(right_table.strip())

        # Add vertical spacing after each row of up to 2 tables
        figure_lines.append("\\vspace{1em}")

    figure_lines.append("\\end{figure}")
    grouped_figures.append("\n".join(figure_lines))

# Output all LaTeX figure blocks
full_latex_output = "\n\n".join(grouped_figures)
print(full_latex_output)


# %% RETURN HISTOGRAM AND TIME SERIES PLOT
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import shared.styling_guidelines_graphs
from shared.styling_guidelines_graphs import colors
import matplotlib.dates as mdates



def plot_return_analysis(df, symbol, return_col='Total Return'):
    df = df.copy()
    df.drop(columns='Set', errors='ignore', inplace=True)

    if 'Symbol' not in df.columns or 'Date' not in df.columns:
        df.reset_index(inplace=True)

    if 'Symbol' not in df.columns:
        raise KeyError(f"'Symbol' column not found. Available columns: {df.columns.tolist()}")

    symbol_df = df[df['Symbol'] == symbol].copy()
    symbol_df = symbol_df.dropna(subset=[return_col])

    # --- Sub-function 1: Time-Series Plot ---
    def plot_time_series(data, symbol, return_col):
        plt.figure(figsize=(7, 4))

        # Make sure Date is datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])

        plt.plot(data['Date'], data[return_col], color='black', linewidth=0.1)

        ax = plt.gca()

        # Set x-axis limits to actual data range
        ax.set_xlim([data['Date'].min(), data['Date'].max()])
        
        # Set y-axis limits to [15,-18]
        ax.set_ylim([-20, 15])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # One tick per year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.title(f"{symbol} Daily Returns")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.tight_layout()
        plt.savefig(f"results/descriptive_statistics/descriptive_statistics_{symbol}_daily_return_plot.pdf")
        plt.show()

    # --- Sub-function 2: Histogram with Scaled Normal Curve ---
    def plot_histogram_with_normal(data, symbol, return_col):
        fig, ax1 = plt.subplots(figsize=(7, 4))

        # Plot histogram (left y-axis)
        sns.histplot(data[return_col], bins=100, stat='frequency',
                    kde=False, color=colors['primary'], edgecolor='black',
                    label="Returns Histogram", ax=ax1, 
                    #make the linewidht thinner 
                    linewidth=0.1
                    )

        ax1.set_xlabel("Return")
        ax1.set_ylabel("Frequency")
        ax1.tick_params(axis='y', labelcolor='black')

        # Add second y-axis (right)
        ax2 = ax1.twinx()

        # Fit normal distribution
        mu, std = norm.fit(data[return_col])
        x = np.linspace(data[return_col].min(), data[return_col].max(), 500)
        p = norm.pdf(x, mu, std)

        # Plot normal PDF (right y-axis)
        ax2.plot(x, p, linestyle='--', color=colors['secondary'], linewidth=2, label='Normal Distrubuted PDF')
        ax2.set_ylabel("Probability Density")
        ax2.tick_params(axis='y', labelcolor='black')

        # Set both y-axes to start from zero
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        # set y-axis limits to [0, 2500] for left y-axis and [0, 0.3] for right y-axis
        ax1.set_ylim(0, 2500)
        ax2.set_ylim(0, 0.35)

        # Set x-axis limits to [-15, 15]
        ax1.set_xlim(-12, 12)   
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))     

        # Combine legends manually
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.title(f"{symbol} Daily Returns Histogram with Normal Curve")
        plt.tight_layout()
        plt.savefig(f"results/descriptive_statistics/descriptive_statistics_{symbol}_daily_return_histogram.pdf")
        plt.show()

    # Call both subplots
    plot_time_series(symbol_df, symbol, return_col)
    plot_histogram_with_normal(symbol_df, symbol, return_col)


# %% Example usage
plot_return_analysis(df_return_eikon, 'AAPL')
plot_return_analysis(df_return_eikon, 'WMT')
# %%
