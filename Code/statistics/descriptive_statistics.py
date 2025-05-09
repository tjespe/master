
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
# Load and preprocess data
processed_data = get_lstm_train_test_new(include_ivol_cols=["Historical Call IVOL", "10 Day Call IVOL"], include_1min_rv=True)

# Extract train, validation, and test datasets
train_data = processed_data.train
validation_data = processed_data.validation
test_data = processed_data.test

# %%
processed_data
# %%


# Import necessary libraries
import pandas as pd
import numpy as np

# Define the descriptive stats function
def descriptive_stats(df):
    stats = pd.DataFrame(index=df.columns)
    stats['Mean'] = df.mean()
    stats['Std'] = df.std()
    stats['Min'] = df.min()
    stats['Median'] = df.median()
    stats['Max'] = df.max()
    stats['Skewness'] = df.skew()
    stats['Kurtosis'] = df.kurtosis()
    return stats

# Drop 'Set' column safely
combined_df = processed_data.df.drop(columns='Set', errors='ignore')

# Group by Symbol and calculate stats per asset
grouped = combined_df.groupby('Symbol')
stats_per_asset = grouped.apply(lambda x: descriptive_stats(x))

# Average the stats across assets
average_stats = stats_per_asset.groupby(stats_per_asset.index.get_level_values(1)).mean()
sds = stats_per_asset.groupby(stats_per_asset.index.get_level_values(1)).std()

# Round for readability
average_stats = average_stats.round(3)
sds = sds.round(3)

# Format average ± std for all stats except Count
# average_stats_formatted = average_stats.astype(str) + " ± " + sds.astype(str)
# average_stats_formatted = average_stats.astype(str) + " {\\scriptsize$\\pm$ " + sds.astype(str) + "}"
# average_stats_formatted = average_stats.round(2).astype(str) + " {\\scriptsize$\\pm$ " + sds.round(2).astype(str) + "}"
average_stats_formatted = average_stats.applymap(lambda x: f"{x:.2f}") + " {\\tiny$\\pm$ " + sds.applymap(lambda x: f"{x:.2f}") + "}"


# Add Count separately
counts = processed_data.df.groupby('Symbol').size().mean().round().astype(int)  # Average number of rows per asset
average_stats_formatted["Count"] = processed_data.df.count()

# Put Count first
cols_order = ["Count", "Mean", "Std", "Min", "Median", "Max", "Skewness", "Kurtosis"]
average_stats_formatted = average_stats_formatted[cols_order]

# # 1. Map raw names to pretty LaTeX names
latex_name_map = {
    "ActualReturn": "Return",
    "RV": "RV\\textsubscript{1-min}",
    "RV_5": "RV\\textsubscript{5-min}",
    "BPV": "BPV\\textsubscript{1-min}",
    "BPV_5": "BPV\\textsubscript{5-min}",
    "Good": "Good\\textsubscript{1-min}",
    "Good_5": "Good\\textsubscript{5-min}",
    "Bad": "Bad\\textsubscript{1-min}",
    "Bad_5": "Bad\\textsubscript{5-min}",
    "RQ": "RQ\\textsubscript{1-min}",
    "RQ_5": "RQ\\textsubscript{5-min}",
    "10 Day Call IVOL": "10 Day Call IVOL",
    "Historical Call IVOL": "Historical Call IVOL",
}

# 2. Your strict manual order
desired_main_order = [
    "ActualReturn", 
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

# Display
print("Table 1: Average Descriptive Statistics Across Assets\n")

# Format as LaTeX
latex_table = average_stats_formatted.to_latex(index=True, escape=False)
print(latex_table)

# %%


# %% 
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



def plot_return_analysis(df, symbol, return_col='ActualReturn'):
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
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data[return_col], color='black', linewidth=0.5)
        plt.title(f"{symbol} Daily Returns")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.tight_layout()
        plt.show()

    # --- Sub-function 2: Histogram with Scaled Normal Curve ---
    def plot_histogram_with_normal(data, symbol, return_col):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot histogram (left y-axis)
        sns.histplot(data[return_col], bins=100, stat='frequency',
                    kde=False, color=colors['primary'], edgecolor='black',
                    label="Returns Histogram", ax=ax1)

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

        # Combine legends manually
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.title(f"{symbol} Daily Returns Histogram with Normal Curve")
        plt.tight_layout()
        plt.show()



    # Call both subplots
    plot_time_series(symbol_df, symbol, return_col)
    plot_histogram_with_normal(symbol_df, symbol, return_col)


# %% Example usage
# Assuming processed_data.df is your main DataFrame
plot_return_analysis(processed_data.df, 'AAPL')
plot_return_analysis(processed_data.df, 'WMT')
# %%
processed_data.df.columns
# %%
