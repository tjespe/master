import numpy as np
from scipy.stats import chi2


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
