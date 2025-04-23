import numpy as np
from scipy.stats import chi2, norm
import pandas as pd
from scipy.optimize import minimize


def christoffersen_test(exceedances, alpha, reset_indices=None):
    """
    exceedances: array-like of 1 if exceedance, 0 otherwise.
    alpha: expected probability of exceedance (e.g. 0.05 for 95% VaR).
    reset_indices: positions in the pooled array that mark the start of a new asset.
    """
    # Ensure positional indexing.
    exceedances = np.asarray(exceedances)
    N = len(exceedances)
    x = np.sum(exceedances)
    pi_hat = x / N

    # Check for degenerate series.
    if x == 0 or x == N:
        return {
            "LR_uc": np.nan,
            "p_value_uc": np.nan,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }

    # Unconditional Coverage Test using logs.
    try:
        logL0 = (N - x) * np.log(1 - alpha) + x * np.log(alpha)
        logL1 = (N - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    except Exception:
        return {
            "LR_uc": np.nan,
            "p_value_uc": np.nan,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }
    LR_uc = -2 * (logL0 - logL1)
    p_value_uc = 1 - chi2.cdf(LR_uc, df=1)

    # Independence Test: use only transitions within each asset.
    valid = np.ones(N, dtype=bool)
    valid[0] = False  # first observation has no predecessor.
    if reset_indices is not None:
        for idx in reset_indices:
            valid[idx] = False

    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) == 0:
        return {
            "LR_uc": LR_uc,
            "p_value_uc": p_value_uc,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }

    # Form transition pairs for valid indices.
    prev = exceedances[valid_indices - 1]
    curr = exceedances[valid_indices]

    N_00 = np.sum((prev == 0) & (curr == 0))
    N_01 = np.sum((prev == 0) & (curr == 1))
    N_10 = np.sum((prev == 1) & (curr == 0))
    N_11 = np.sum((prev == 1) & (curr == 1))

    denom_0 = N_00 + N_01
    denom_1 = N_10 + N_11
    pi_0 = N_01 / denom_0 if denom_0 > 0 else 0
    pi_1 = N_11 / denom_1 if denom_1 > 0 else 0

    # If either conditional probability is degenerate, skip the independence test.
    if pi_0 in [0, 1] or pi_1 in [0, 1]:
        LR_ind = np.nan
        p_value_ind = np.nan
    else:
        try:
            logL_null = (N - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
            logL_alt = (
                N_00 * np.log(1 - pi_0)
                + N_01 * np.log(pi_0)
                + N_10 * np.log(1 - pi_1)
                + N_11 * np.log(pi_1)
            )
        except Exception:
            LR_ind = np.nan
            p_value_ind = np.nan
        else:
            LR_ind = -2 * (logL_null - logL_alt)
            p_value_ind = 1 - chi2.cdf(LR_ind, df=1)

    LR_cc = LR_uc + LR_ind if (not np.isnan(LR_uc) and not np.isnan(LR_ind)) else np.nan
    p_value_cc = 1 - chi2.cdf(LR_cc, df=2) if not np.isnan(LR_cc) else np.nan

    return {
        "LR_uc": LR_uc,
        "p_value_uc": p_value_uc,
        "LR_ind": LR_ind,
        "p_value_ind": p_value_ind,
        "LR_cc": LR_cc,
        "p_value_cc": p_value_cc,
    }


def joint_esr_loss(params, y, var_forecast, es_forecast, alpha):
    """
    Joint loss function for quantile (VaR) and ES regression, from Fissler & Ziegel (2016).
    """
    b1, b2, c1, c2 = params
    q = b1 + b2 * var_forecast
    e = c1 + c2 * es_forecast
    indicator = (y <= q).astype(float)
    loss = (1 / alpha) * ((indicator * (q - y)) - (q - e)) + np.log(e)
    return np.mean(loss)


def estimate_joint_esr(y, var_forecast, es_forecast, alpha):
    # Initial guess for [b1, b2, c1, c2]
    init_params = np.array([0.0, 1.0, 0.0, 1.0])
    result = minimize(
        joint_esr_loss,
        init_params,
        args=(y, var_forecast, es_forecast, alpha),
        method="BFGS",
    )
    return result.x, result.hess_inv


def auxiliary_esr_test(y, var_forecast, es_forecast, alpha):
    T = len(y)
    params, cov = estimate_joint_esr(y, var_forecast, es_forecast, alpha)
    c1, c2 = params[2], params[3]
    cov_c = cov[2:, 2:]
    test_stat = T * np.dot(np.dot([c1, c2 - 1], np.linalg.inv(cov_c)), [c1, c2 - 1])
    p_value = 1 - chi2.cdf(test_stat, df=2)
    return test_stat, p_value


def pooled_bayer_dimitriadis_test(y_true, var_pred, es_pred, alpha):
    """
    Test that the expected value of VaR exceedances matches the expected shortfall,
    i.e. that
    E[ I(y_t > VaR_t^alpha) * (y_t - ES_t^alpha) ] = 0
    where I is the indicator function.

    This version pools the test variable z_t across all assets before computing the test statistic.

    Parameters:
        y_true   : (N, T) array-like of actual losses where N = # of assets and T = # of observations
        var_pred : (N, T) array-like of VaR predictions where N = # of assets and T = # of observations
        es_pred  : (N, T) array-like of ES predictions where N = # of assets and T = # of observations
        alpha    : VaR level
    """
    # Convert inputs to np.array for safety
    y_true = np.asarray(y_true)
    var_pred = np.asarray(var_pred)
    es_pred = np.asarray(es_pred)

    n = len(y_true)
    if any(len(arr) != n for arr in [var_pred, es_pred]):
        raise ValueError("y_true, var_pred, es_pred must have the same length.")

    # Indicator of exceedance: 1 if actual loss is bigger than the VaR threshold
    exceedances = y_true < var_pred

    # Define the test variable z_t
    # z_t = I(X_t > VaR_t^alpha) * [ (X_t - ES_t^alpha) / alpha ]
    z = np.where(exceedances, (y_true - es_pred) / alpha, 0.0).mean(axis=0)

    mean_z = np.nanmean(z)
    std_z = np.nanstd(z, ddof=1)

    # If the test variable is degenerate, return NaNs
    if std_z == 0:
        print("SD[Z] was 0 for alpha", alpha, "returning NaNs")
        print("mean_z", mean_z)
        print("std_z", std_z)
        # print("exceedances", exceedances)
        print("y_true", y_true)
        print("var_pred", var_pred)
        print("es_pred", es_pred)
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "mean_z": mean_z,
            "std_z": std_z,
        }

    # Compute test statistic: T = sqrt(n) * (mean of z) / stdev(z)
    test_statistic = np.sqrt(n) * mean_z / std_z
    # Two-sided p-value
    p_value = 2.0 * (1.0 - norm.cdf(abs(test_statistic)))

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "mean_z": mean_z,
        "std_z": std_z,
    }
