# File for enriching prediction files before ES testing in R

# %%
# Load libraries
from scipy.stats import norm
import pandas as pd
import numpy as np

from settings import DATA_PATH, TEST_SET, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT
from shared.mdn import calculate_es_for_quantile
from shared.conf_levels import format_cl

CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]
# %%
# HAR models
har_preds_orig = pd.read_csv("predictions/HAR_R.csv")
har_preds_orig["Date"] = pd.to_datetime(har_preds_orig["Date"])
har_preds = har_preds_orig.set_index(["Date", "Symbol"])
har_dates = har_preds.index.get_level_values("Date")
har_preds = har_preds[
    (
        (har_dates >= TRAIN_VALIDATION_SPLIT)
        & (har_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (har_dates >= VALIDATION_TEST_SPLIT)
    )
]

har_vol_preds = har_preds["HAR_vol_R"].values
mus = np.zeros_like(har_vol_preds)

for cl in CONFIDENCE_LEVELS:
    alpha = 1 - cl
    z_alpha = norm.ppf(1 - alpha / 2)
    lb = mus - z_alpha * har_vol_preds
    ub = mus + z_alpha * har_vol_preds
    har_preds[f"LB_{format_cl(cl)}"] = lb
    har_preds[f"UB_{format_cl(cl)}"] = ub
    es_alpha = alpha / 2
    har_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
        np.ones_like(mus).reshape(-1, 1),
        mus.reshape(-1, 1),
        har_vol_preds.reshape(-1, 1),
        lb,
    )
# %%
# write enriched HAR predictions back to the same file

har_preds.reset_index(inplace=True)
har_preds.to_csv("predictions/HAR_R.csv", index=False)



# %%
# HARQ model
harq_preds_orig = pd.read_csv("predictions/HARQ_R.csv")
harq_preds_orig["Date"] = pd.to_datetime(harq_preds_orig["Date"])
harq_preds = harq_preds_orig.set_index(["Date", "Symbol"])
harq_dates = harq_preds.index.get_level_values("Date")
harq_preds = harq_preds[
    (
        (harq_dates >= TRAIN_VALIDATION_SPLIT)
        & (harq_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (harq_dates >= VALIDATION_TEST_SPLIT)
    )
]
harq_vol_preds = harq_preds["HARQ_vol_R"].values
mus = np.zeros_like(harq_vol_preds)

for cl in CONFIDENCE_LEVELS:
    alpha = 1 - cl
    z_alpha = norm.ppf(1 - alpha / 2)
    lb = mus - z_alpha * harq_vol_preds
    ub = mus + z_alpha * harq_vol_preds
    harq_preds[f"LB_{format_cl(cl)}"] = lb
    harq_preds[f"UB_{format_cl(cl)}"] = ub
    es_alpha = alpha / 2
    harq_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
        np.ones_like(mus).reshape(-1, 1),
        mus.reshape(-1, 1),
        harq_vol_preds.reshape(-1, 1),
        lb,
    )
# %%
# write enriched HARQ predictions back to the same file
harq_preds.reset_index(inplace=True)
harq_preds.to_csv("predictions/HARQ_R.csv", index=False)


# %%
# GARCH
data = pd.read_csv("data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")
# keep coloumns Date, Symbol, GARCH_Vol and GARCH_Mean
garch_preds_orig = data[["Date", "Symbol", "GARCH_Vol", "GARCH_Mean"]]
garch_preds_orig["Date"] = pd.to_datetime(garch_preds_orig["Date"])
garch_preds = garch_preds_orig.set_index(["Date", "Symbol"])
garch_dates = garch_preds.index.get_level_values("Date")
garch_preds = garch_preds[
    (
        (garch_dates >= TRAIN_VALIDATION_SPLIT)
        & (garch_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (garch_dates >= VALIDATION_TEST_SPLIT)
    )
]
garch_vol_preds = garch_preds["GARCH_Vol"].values
mus = garch_preds["GARCH_Mean"].values

for cl in CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * garch_vol_preds
            ub = mus + z_alpha * garch_vol_preds
            garch_preds[f"LB_{format_cl(cl)}"] = lb
            garch_preds[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            garch_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                garch_vol_preds.reshape(-1, 1),
                lb,
            )

# %%
# write enriched GARCH predictions to new file
garch_preds.reset_index(inplace=True)
garch_preds.to_csv("predictions/GARCH_preds_enriched.csv", index=False)


# %%
# EGARCH
data = pd.read_csv("data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")
# keep coloumns Date, Symbol, EGARCH_Vol and EGARCH_Mean
egarch_preds_orig = data[["Date", "Symbol", "EGARCH_Vol", "EGARCH_Mean"]]
egarch_preds_orig["Date"] = pd.to_datetime(egarch_preds_orig["Date"])
egarch_preds = egarch_preds_orig.set_index(["Date", "Symbol"])
egarch_dates = egarch_preds.index.get_level_values("Date")
egarch_preds = egarch_preds[
    (
        (egarch_dates >= TRAIN_VALIDATION_SPLIT)
        & (egarch_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (egarch_dates >= VALIDATION_TEST_SPLIT)
    )
]
egarch_vol_preds = egarch_preds["EGARCH_Vol"].values
mus = egarch_preds["EGARCH_Mean"].values

for cl in CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * egarch_vol_preds
            ub = mus + z_alpha * egarch_vol_preds
            egarch_preds[f"LB_{format_cl(cl)}"] = lb
            egarch_preds[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            egarch_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                egarch_vol_preds.reshape(-1, 1),
                lb,
            )
# %%
# write enriched EGARCH predictions to new file
egarch_preds.reset_index(inplace=True)
egarch_preds.to_csv("predictions/EGARCH_preds_enriched.csv", index=False)



# %%
# RV GARCH
rv_GARCH_preds_orig = pd.read_csv("predictions/realized_garch_forecast_norm.csv")
rv_GARCH_preds_orig["Date"] = pd.to_datetime(rv_GARCH_preds_orig["Date"])
rv_GARCH_preds = rv_GARCH_preds_orig.set_index(["Date", "Symbol"])
rv_GARCH_dates = rv_GARCH_preds.index.get_level_values("Date")
rv_GARCH_preds = rv_GARCH_preds[
    (
        (rv_GARCH_dates >= TRAIN_VALIDATION_SPLIT)
        & (rv_GARCH_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (rv_GARCH_dates >= VALIDATION_TEST_SPLIT)
    )
]
rv_GARCH_vol_preds = rv_GARCH_preds["Forecast_Volatility"].values
mus = rv_GARCH_preds["Mean"].values

for cl in CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * rv_GARCH_vol_preds
            ub = mus + z_alpha * rv_GARCH_vol_preds
            rv_GARCH_preds[f"LB_{format_cl(cl)}"] = lb
            rv_GARCH_preds[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            rv_GARCH_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                rv_GARCH_vol_preds.reshape(-1, 1),
                lb,
            )
# %%
# write enriched RV GARCH predictions to existing file
rv_GARCH_preds.reset_index(inplace=True)
rv_GARCH_preds.to_csv("predictions/realized_garch_forecast_norm.csv", index=False)