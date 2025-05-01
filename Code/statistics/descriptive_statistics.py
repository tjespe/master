
#%% Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys

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
