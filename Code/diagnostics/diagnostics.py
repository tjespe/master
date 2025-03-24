#### File implementing Diagnostc tests on the data ####


# Should implement the following tests:
# 1. Augmented Dikcy Fuller test - for stationarity (test on each stock)
# 2. Ljung-Box or Breusch-Godfrey for serial correlation (test on each stock)
# 3. ARCH-LM (Engle) test for conditional heteroskedasticity (test on each stock)
# 4. Durbin-Watson statistic for residual autocorrelation (test on each stock) (post-diagnostic after a Model is made)
# 5. Jarque-Bera test for normality (test on each stock)
# 6 VIF (Variance Inflation Factor) for multicollinearity (test on each stock)


# %%
# import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import reset_ramsey
from tqdm import tqdm

# %%
# Load the data
df_returns = pd.read_csv("../data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")
df_capire = pd.read_csv("../data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")

# do some cleaning
df_returns["Symbol"] = df_returns["Symbol"].str.replace(".O", "")
df_returns["Date"] = pd.to_datetime(df_returns["Date"])
df_returns = df_returns.sort_values(["Symbol", "Date"])
df_returns["Total Return"] = df_returns["Total Return"] / 100
df_returns.reset_index(drop=True, inplace=True)

df_capire = df_capire.sort_values(["Symbol", "Date"])
df_capire["Date"] = pd.to_datetime(df_capire["Date"])

# merge data    
df = pd.merge(df_returns, df_capire, on=["Date", "Symbol"], how="inner")
df = df.set_index(["Date", "Symbol"])
df


# %%
# 1. Augmented Dickey Fuller test - for stationarity (test on each stock)
def adf_test(series):
    result = adfuller(series)
    return result[1] < 0.05

# apply the test on each stock
adf_results = df.groupby("Symbol")["LogReturn"].apply(adf_test)
# dsiplay the results as a dataframe
adf_results = pd.DataFrame(adf_results)
adf_results.columns = ["Stationary"]
adf_results

# %% - alternative implementation
def adf_test(series, symbol, regression='c'):
    """
    Runs the ADF test and returns a dictionary of results.
    """
    series = series.dropna()
    
    if len(series) < 10:
        # Too few observations to run the test reliably
        return {
            'Symbol': symbol,
            'ADF Statistic': None,
            'p-value': None,
            'Num Lags Used': None,
            'Num Observations Used': len(series),
            'Critical Value 1%': None,
            'Critical Value 5%': None,
            'Critical Value 10%': None,
            'Stationary?': 'Insufficient data'
        }
    
    result = adfuller(series, regression=regression, autolag='AIC')
    
    return {
        'Symbol': symbol,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Num Lags Used': result[2],
        'Num Observations Used': result[3],
        'Critical Value 1%': result[4]['1%'],
        'Critical Value 5%': result[4]['5%'],
        'Critical Value 10%': result[4]['10%'],
        'Stationary?': 'Yes' if result[1] < 0.05 else 'No'
    }

# Initialize an empty list to store the results
adf_results = []	
# Group by the symbol and apply the ADF test
for symbol, group in df_returns.groupby('Symbol'):
    returns_series = group['LogReturn']
    # Run ADF test for this symbol's returns
    result = adf_test(returns_series, symbol)
    # Append the results
    adf_results.append(result)

# Convert the list of dictionaries into a DataFrame
adf_results_df = pd.DataFrame(adf_results)

# Display the results
adf_results_df

# %%
# 2. Ljung-Box test for serial correlation (test on each stock)
def ljung_box_test(series):
    result = acorr_ljungbox(series, lags=20, return_df=True)
    p_value = result.iloc[-1]["lb_pvalue"]
    return p_value < 0.05

# apply the test on each stock
ljung_box_results = df.groupby("Symbol")["LogReturn"].apply(ljung_box_test)
# dsiplay the results as a dataframe
ljung_box_results = pd.DataFrame(ljung_box_results)
ljung_box_results.columns = ["Serial Correlation"]

ljung_box_results


# do the same with SquredReturn
ljung_box_results_squared = df.groupby("Symbol")["SquaredReturn"].apply(ljung_box_test)
# dsiplay the results as a dataframe
ljung_box_results_squared = pd.DataFrame(ljung_box_results_squared)
ljung_box_results_squared.columns = ["Serial Correlation Squared"]

ljung_box_results_squared
# merge the two dataframes
ljung_box_results_final = pd.merge(ljung_box_results, ljung_box_results_squared, on="Symbol")
ljung_box_results_final

# %% - alternative implementatin
def ljung_box_test(series, symbol, column_name, lags=10):
    """
    Runs Ljung-Box test on any time series column.

    Args:
        series (pd.Series): The time series data (already returns or squared returns).
        symbol (str): Stock symbol.
        column_name (str): Name of the column being tested (for reporting).
        lags (int): Number of lags to test.

    Returns:
        dict: Ljung-Box test results (p-value at max lag).
    """
    series = series.dropna()

    if len(series) < (lags + 1):
        return {
            'Symbol': symbol,
            'Column': column_name,
            'Lags': lags,
            'LB Stat': None,
            'p-value': None,
            'Autocorrelation?': 'Insufficient data'
        }

    # Perform Ljung-Box test (no need to square here)
    lb_test = acorr_ljungbox(series, lags=lags, return_df=True)

    lb_stat = lb_test.iloc[-1]['lb_stat']
    p_value = lb_test.iloc[-1]['lb_pvalue']

    return {
        'Symbol': symbol,
        'Column': column_name,
        'Lags': lags,
        'LB Stat': lb_stat,
        'p-value': p_value,
        'Autocorrelation?': 'Yes' if p_value < 0.05 else 'No'
    }


# Collect results here
ljungbox_results = []

# Set number of lags for testing
lags = 10

# Loop through each stock symbol
for symbol, group in df_returns.groupby('Symbol'):
    # Run Ljung-Box test on Returns
    result_returns = ljung_box_test(group['LogReturn'], symbol, 'Returns', lags=lags)
    ljungbox_results.append(result_returns)
    
    # Run Ljung-Box test on Squared Returns (ARCH effects)
    result_squared_returns = ljung_box_test(group['SquaredReturn'], symbol, 'Squared Returns', lags=lags)
    ljungbox_results.append(result_squared_returns)

# Convert results to DataFrame
ljungbox_results_df = pd.DataFrame(ljungbox_results)

ljungbox_results_df

# %%
# 3. ARCH-LM (Engle) test for conditional heteroskedasticity (test on each stock)
def arch_lm_test(series):
    result = het_arch(series, nlags=20)
    p_value = result[1]
    return p_value < 0.05

# apply the test on each stock
arch_lm_results = df.groupby("Symbol")["LogReturn"].apply(arch_lm_test)
# dsiplay the results as a dataframe
arch_lm_results = pd.DataFrame(arch_lm_results)
arch_lm_results.columns = ["Conditional Heteroskedasticity"]
arch_lm_results



# %%
# 5. Jarque-Bera test for normality (test on each stock)
def jarque_bera_test(series):
    result = jarque_bera(series)
    return result[1] > 0.05

# apply the test on each stock
jarque_bera_results = df.groupby("Symbol")["LogReturn"].apply(jarque_bera_test)
# dsiplay the results as a dataframe
jarque_bera_results = pd.DataFrame(jarque_bera_results)
jarque_bera_results.columns = ["Normality"]
jarque_bera_results

# %%
