# %%
# Settings
FNAME = "predictions/realized_garch_forecast_std.csv"
MEAN_COL = "Mean"
VOL_COL = "Forecast_Volatility"

# %%
# CD to parent directory
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

# %%
# Imports
import pandas as pd
import numpy as np

from shared.conf_levels import format_cl
from shared.mdn import calculate_es_for_quantile
from scipy.stats import norm


# %%
# load the dataframe
results_df = pd.read_csv(FNAME)

# %%
# Add VaR and ES estimates assuming normality
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]
ALL_CONFIDENCE_LEVELS = CONFIDENCE_LEVELS + [0.99, 0.995]
mu = results_df[MEAN_COL].values
har_vol_pred = results_df[VOL_COL].values
for cl in ALL_CONFIDENCE_LEVELS:
    alpha = 1 - cl
    z_alpha = norm.ppf(1 - alpha / 2)
    lb = mu - z_alpha * har_vol_pred
    ub = mu + z_alpha * har_vol_pred
    results_df[f"LB_{format_cl(cl)}"] = lb
    results_df[f"UB_{format_cl(cl)}"] = ub
    es_alpha = alpha / 2
    results_df[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
        np.ones_like(mu).reshape(-1, 1),
        mu.reshape(-1, 1),
        har_vol_pred.reshape(-1, 1),
        lb,
    )

# %%
# Save the dataframe with the new columns
results_df.to_csv(FNAME, index=False)
