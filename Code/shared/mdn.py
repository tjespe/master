import gc
import math
import numpy as np
from numba import njit
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.stats import norm

from settings import TEST_ASSET
from shared.numerical_mixture_moments import numerical_mixture_moments


def get_mdn_kernel_initializer(n_mixtures):
    """
    Returns a custom initializer for the MDN output layer.
    The weight matrix is divided into three parts (columns):
      - First n_mixtures: for logits (initialized with GlorotUniform)
      - Second n_mixtures: for mu's (initialized with RandomNormal with stddev=0.01)
      - Third n_mixtures: for log-variances (initialized with GlorotUniform)
    """

    def mdn_kernel_initializer(shape, dtype=None, partition_info=None):
        # Expecting shape = (input_dim, 3 * n_mixtures)
        input_dim, total_units = shape
        if total_units != 3 * n_mixtures:
            raise ValueError("The output dimension does not equal 3 * n_mixtures.")

        # Initialize the three parts:
        init_logits = tf.keras.initializers.RandomNormal(
            mean=1 / n_mixtures, stddev=1 / (n_mixtures * 3)
        )(shape=(input_dim, n_mixtures), dtype=dtype)
        init_mu = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)(
            shape=(input_dim, n_mixtures), dtype=dtype
        )
        init_logvar = tf.keras.initializers.RandomNormal(mean=0.0, stddev=2)(
            shape=(input_dim, n_mixtures), dtype=dtype
        )

        # Concatenate them along the last axis (columns)
        kernel = tf.concat([init_logits, init_mu, init_logvar], axis=-1)
        return kernel

    return mdn_kernel_initializer


def get_mdn_bias_initializer(n_mixtures, logvar_bias=-9):
    """
    Custom initializer for the MDN output layer's bias.
    The bias vector is of shape (3 * n_mixtures,). We set:
      - The first n_mixtures (logits) to 0,
      - The next n_mixtures (mu) to 0,
      - The last n_mixtures (log-variance) to logvar_bias.
    """

    def mdn_bias_initializer(shape, dtype=None, partition_info=None):
        if shape[0] != 3 * n_mixtures:
            raise ValueError("Bias shape must be (3 * n_mixtures,)")
        bias = np.zeros((3 * n_mixtures,), dtype=np.float32)
        bias[2 * n_mixtures :] = logvar_bias
        return tf.convert_to_tensor(bias, dtype=dtype)

    return mdn_bias_initializer


def parse_mdn_output(mdn_out, n_mixtures):
    """
    Given y_pred from the model with shape (batch, 3*n_mixtures),
    parse out pi, mu, log_var. Returns (pi, mu, sigma) each shape = (batch, n_mixtures).
    """
    logits_pi = mdn_out[:, :n_mixtures]
    mu = mdn_out[:, n_mixtures : 2 * n_mixtures]
    log_var = mdn_out[:, 2 * n_mixtures :]

    pi = tf.nn.softmax(logits_pi, axis=-1)
    sigma = tf.exp(0.5 * log_var)

    return pi, mu, sigma


def univariate_mixture_mean_and_var_approx(pi, mu, sigma_):
    """
    Creates a univariate approximation of the mixture distribution.

    For univariate mixture:
      mixture_mean = sum(pi_k * mu_k)
      mixture_var  = sum(pi_k * (sigma_k^2 + mu_k^2)) - mixture_mean^2
    """
    mixture_mean = tf.reduce_sum(pi * mu, axis=1)
    mixture_mean_sq = tf.square(mixture_mean)
    # Avoid blowing up result when a mixture has very high sigma
    # (this means it has a very low weight and don't really matter to the distribution,
    # but since we square the sigma below it can affect the result)
    sigma = tf.minimum(sigma_, 1)
    # E[sigma^2 + mu^2] = sum_k(pi_k * (sigma_k^2 + mu_k^2))
    e_sigma2_mu2 = tf.reduce_sum(pi * (tf.square(sigma) + tf.square(mu)), axis=1)
    mixture_var = e_sigma2_mu2 - mixture_mean_sq
    return mixture_mean, mixture_var


# Example MC Dropout approach specialized for MDN outputs
def predict_with_mc_dropout_mdn(model, X, T=100, n_mixtures=5):
    """
    Performs T stochastic forward passes with model(X, training=True).
    Aggregates mixture distribution => splits aleatoric vs. epistemic.

    Returns a dict with:
      "expected_returns" : mean of mixture means across MC samples
      "volatility_estimates" : sqrt of total variance (aleatoric + epistemic)
      "epistemic_uncertainty_expected_returns" : std dev of mixture means across MC
      "epistemic_uncertainty_volatility_estimates": see notes
      ...
    Feel free to adapt as needed.
    """
    # Storage for mixture means, mixture variances across T samples
    mc_means = []
    mc_vars = []

    # Run T forward passes
    for _ in range(T):
        mdn_out = model(X, training=True)  # (batch, 3*n_mixtures)
        pi, mu, sigma = parse_mdn_output(mdn_out, n_mixtures)

        # compute mixture mean & var for each sample
        mean_s, var_s = univariate_mixture_mean_and_var_approx(pi, mu, sigma)
        mc_means.append(np.array(mean_s))
        mc_vars.append(np.array(var_s))

    mc_means = np.array(mc_means)  # shape: (T, batch)
    mc_vars = np.array(mc_vars)  # shape: (T, batch)

    # Aleatoric = average of variances
    aleatoric_var = np.mean(mc_vars, axis=0)  # (batch,)

    # Epistemic = variance of means across MC
    epistemic_var = np.var(mc_means, axis=0)  # (batch,)

    # Final predicted mean = average of mixture means
    final_means = np.mean(mc_means, axis=0)

    # total variance = aleatoric + epistemic
    total_var = aleatoric_var + epistemic_var

    results = {
        "expected_returns": final_means,  # shape (batch,)
        "volatility_estimates": np.sqrt(total_var),
        "epistemic_uncertainty_expected_returns": np.sqrt(epistemic_var),
        "epistemic_uncertainty_volatility_estimates": np.sqrt(epistemic_var),
        # or define your own decomposition
    }
    return results


def compute_mixture_pdf(x_vals, pi, mu, sigma):
    """
    Compute the PDF of a mixture of normals at given x values.

    Parameters:
      x_vals : np.ndarray
          1D array of x values at which to evaluate the PDF.
      pi : np.ndarray
          1D array of mixture weights (should sum to 1).
      mu : np.ndarray
          1D array of means for each mixture component.
      sigma : np.ndarray
          1D array of standard deviations for each mixture component.

    Returns:
      np.ndarray: PDF values evaluated at x_vals.
    """
    # Ensure arrays are numpy arrays in case they're not already.
    pi, mu, sigma = np.asarray(pi), np.asarray(mu), np.asarray(sigma)

    # Evaluate each component's pdf; shape will be (n_components, len(x_vals))
    component_pdfs = (1 / (sigma[:, None] * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_vals - mu[:, None]) / sigma[:, None]) ** 2
    )

    # Weight each component and sum over components.
    return np.sum(pi[:, None] * component_pdfs, axis=0)


def calculate_intervals(pis, mus, sigmas, confidence_levels):
    """
    Calculate the (lower, upper) quantile intervals for a Gaussian mixture for several
    confidence levels simultaneously.

    For each sample i, the mixture CDF is defined as:
        F_i(x) = sum_m pis[i, m] * Phi((x - mus[i, m]) / sigmas[i, m])
    where Phi is the standard normal CDF. For each confidence level cl, the interval is
    determined by:
        lower_i  such that F_i(lower_i) = (1 - cl)/2
        upper_i  such that F_i(upper_i) = 1 - (1 - cl)/2

    Args:
        pis: Mixture weights, shape (n_samples, n_mixtures)
        mus: Mixture means, shape (n_samples, n_mixtures)
        sigmas: Mixture standard deviations, shape (n_samples, n_mixtures)
        confidence_levels: Iterable of confidence levels (floats), e.g., [0.95, 0.5]

    Returns:
        intervals: Array of shape (n_samples, n_confidence_levels, 2) where
            intervals[i, j, 0] is the lower bound and intervals[i, j, 1] is the upper bound
            for sample i at confidence level confidence_levels[j].
    """
    pis = np.asarray(pis)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    conf_levels = np.asarray(confidence_levels)

    n_samples, n_mixtures = pis.shape
    n_conf = conf_levels.size
    intervals = np.empty((n_samples, n_conf, 2))

    # To bracket all quantile roots for every cl, we use the most extreme targets,
    # which come from the highest confidence level.
    max_conf = np.max(conf_levels)
    global_alpha = (1 - max_conf) / 2  # smallest quantile target (further left)
    global_beta = 1 - global_alpha  # largest quantile target (further right)

    for i in range(n_samples):
        weights = pis[i, :]
        means = mus[i, :]
        stds = sigmas[i, :]

        # Mixture CDF for sample i.
        def mixture_cdf(x):
            return np.sum(weights * norm.cdf((x - means) / stds))

        # Set an initial search interval.
        s = np.max(stds)
        low = -1
        high = 1

        # Expand search interval if necessary.
        while mixture_cdf(low) > global_alpha:
            low -= 10 * s
        while mixture_cdf(high) < global_beta:
            high += 10 * s

        # Compute intervals for each confidence level.
        for j, cl in enumerate(conf_levels):
            alpha = (1 - cl) / 2
            beta = 1 - alpha  # equivalently, (1 + cl) / 2
            lower_bound = brentq(lambda x: mixture_cdf(x) - alpha, low, high)
            upper_bound = brentq(lambda x: mixture_cdf(x) - beta, low, high)
            intervals[i, j, 0] = lower_bound
            intervals[i, j, 1] = upper_bound

    return intervals


def plot_sample_days(
    y_dates: list[str],
    y_test: np.ndarray,
    pi_pred: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    n_mixtures: int,
    save_to: Optional[str] = None,
    show=True,
    ticker=TEST_ASSET,
):
    plt.figure(figsize=(10, 40))
    np.random.seed(0)
    days = np.random.randint(0, len(y_test), 10)
    days = np.sort(days)[::-1]
    for i, day in enumerate(days):
        plt.subplot(10, 1, i + 1)
        timestamp = y_dates[-day]
        x_min = -0.1
        x_max = 0.1
        x_vals = np.linspace(x_min, x_max, 1000)
        mixture_pdf = compute_mixture_pdf(
            x_vals, pi_pred[-day], mu_pred[-day], sigma_pred[-day]
        )
        plt.fill_between(
            x_vals,
            np.zeros_like(x_vals),
            mixture_pdf,
            color="blue",
            label="Mixture",
            alpha=0.5,
        )
        plotted_mixtures = 0
        top_weights = np.argsort(pi_pred[-day])[-7:][::-1]
        for j in range(n_mixtures):
            weight = np.array(pi_pred[-day, j])
            if weight < 0.001:
                continue
            plotted_mixtures += 1
            mu = mu_pred[-day, j]
            sigma = sigma_pred[-day, j]
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_vals - mu) / sigma) ** 2
            )
            legend = f"$\pi_{{{j}}}$ = {weight*100:.2f}%" if j in top_weights else None
            plt.plot(x_vals, pdf, label=legend, alpha=min(10 * weight, 1))
        plt.axvline(y_test[-day], color="red", linestyle="--", label="Actual")
        moment_estimates = numerical_mixture_moments(
            np.array(pi_pred[-day]),
            np.array(mu_pred[-day]),
            np.array(sigma_pred[-day]),
            range_factor=3,
        )
        plt.axvline(
            moment_estimates["mean"],
            color="black",
            linestyle="--",
            label="Predicted Mean",
        )
        plt.text(
            x_min + 0.01,
            5,
            f"Mean: {moment_estimates['mean']*100:.2f}%\n"
            f"Std: {moment_estimates['std']*100:.2f}%\n"
            f"Skewness: {moment_estimates['skewness']:.4f}*\n"
            f"Excess kurtosis: {moment_estimates['excess_kurtosis']:.4f}*\n"
            f"* Numerically estimated",
            fontsize=10,
        )
        plt.gca().set_xticklabels(
            ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
        )
        plt.title(
            f"{timestamp.strftime('%Y-%m-%d')} - Predicted Return Distribution for {ticker}"
        )
        plt.ylim(0, 50)
        plt.legend()
        plt.ylabel("Density")
    plt.xlabel("LogReturn")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()


@njit
def rowwise_max_2d(arr):
    """
    Return the maximum of each row in a 2D array.
    Equivalent to np.max(arr, axis=1) but JIT-friendly.
    """
    n_rows, n_cols = arr.shape
    out = np.empty(n_rows, dtype=arr.dtype)
    for i in range(n_rows):
        m = arr[i, 0]
        for j in range(1, n_cols):
            if arr[i, j] > m:
                m = arr[i, j]
        out[i] = m
    return out


@njit
def mixture_cdf(x, pis, mus, sigmas):
    """
    Compute mixture CDF at x for all samples (n_samples).
    x: shape (n_samples,)
    pis, mus, sigmas: shape (n_samples, n_mixtures)
    """
    n_samples, n_mixtures = pis.shape
    out = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        cdf_val = 0.0
        for m in range(n_mixtures):
            z = (x[i] - mus[i, m]) / sigmas[i, m]
            # Use an erf-based expression for normal CDF
            cdf_val += pis[i, m] * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        out[i] = cdf_val
    return out


@njit
def bracket_and_bisect(pis, mus, sigmas, alpha, max_iter=50, tol=1e-8):
    """
    Vectorized bracket + bisection with Numba JIT.
    Finds x_i s.t. mixture_cdf(x_i) = alpha for each sample i.
    """
    n_samples = pis.shape[0]

    # Instead of np.max(sigmas, axis=1), use the helper:
    s = rowwise_max_2d(sigmas)

    low_vals = np.full(n_samples, -1.0, dtype=np.float64)
    high_vals = np.full(n_samples, 1.0, dtype=np.float64)

    f_low = mixture_cdf(low_vals, pis, mus, sigmas)
    # Expand bracket on the left
    done_mask = np.zeros(n_samples, dtype="bool")  # which ones are done expanding
    while True:
        still_too_high = (f_low > alpha) & (~done_mask)
        if not np.any(still_too_high):
            break
        for i in range(n_samples):
            if still_too_high[i]:
                low_vals[i] -= 10.0 * s[i]
                f_low[i] = mixture_cdf(
                    np.array([low_vals[i]]),
                    pis[i : i + 1],
                    mus[i : i + 1],
                    sigmas[i : i + 1],
                )[0]
                # If it's still not low enough, we'll catch it in next iteration
            else:
                done_mask[i] = True

    f_high = mixture_cdf(high_vals, pis, mus, sigmas)
    # Expand bracket on the right
    done_mask[:] = False
    while True:
        still_too_low = (f_high < alpha) & (~done_mask)
        if not np.any(still_too_low):
            break
        for i in range(n_samples):
            if still_too_low[i]:
                high_vals[i] += 10.0 * s[i]
                f_high[i] = mixture_cdf(
                    np.array([high_vals[i]]),
                    pis[i : i + 1],
                    mus[i : i + 1],
                    sigmas[i : i + 1],
                )[0]
            else:
                done_mask[i] = True

    # Bisection
    for it in range(max_iter):
        mid_vals = 0.5 * (low_vals + high_vals)
        f_mid = mixture_cdf(mid_vals, pis, mus, sigmas)
        for i in range(n_samples):
            if f_mid[i] < alpha:
                low_vals[i] = mid_vals[i]
            else:
                high_vals[i] = mid_vals[i]
        # Early stop if intervals are tight
        if (high_vals - low_vals).max() < tol:
            print("Converged after", it + 1, "iterations")
            break
    else:
        print("Warning: Bisection did not converge after", max_iter, "iterations")
        print("Max high low diff:", (high_vals - low_vals).max())

    return 0.5 * (low_vals + high_vals)


def calculate_intervals_vectorized(pis, mus, sigmas, confidence_levels):
    """
    Vectorized calculation of (lower, upper) quantiles for each sample and
    each confidence level.
    """
    pis = np.asarray(pis)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    conf_levels = np.asarray(confidence_levels)

    n_samples = pis.shape[0]
    n_conf = conf_levels.size
    intervals = np.empty((n_samples, n_conf, 2))

    for j, cl in enumerate(conf_levels):
        alpha = (1 - cl) / 2
        beta = 1 - alpha

        # Solve mixture_cdf(x) = alpha and mixture_cdf(x) = beta in parallel
        print("Calculating lower bounds for", cl, "i.e. q =", alpha)
        lower_bound = bracket_and_bisect(pis, mus, sigmas, alpha)
        print("Calculating upper bounds for", cl, "i.e. q =", beta)
        upper_bound = bracket_and_bisect(pis, mus, sigmas, beta)

        intervals[:, j, 0] = lower_bound
        intervals[:, j, 1] = upper_bound

    return intervals


def calculate_prob_above_zero_vectorized(pis, mus, sigmas):
    """
    Vectorized calculation of the probability that a mixture distribution
    is above zero, for each row (sample) in pis, mus, sigmas.
    """
    pis = np.asarray(pis)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # Standard normal CDF at -mu/sigma
    cdf_vals = norm.cdf(-mus / sigmas)

    # Mixture CDF at zero = sum_i pi_i * cdf(-mu_i/sigma_i)
    mixture_cdf_zero = np.sum(pis * cdf_vals, axis=1)

    # Probability above zero
    return 1 - mixture_cdf_zero


def calculate_es_for_quantile(pis, mus, sigmas, var_values):
    """
    Vectorized computation of Expected Shortfall (ES).

    Parameters:
    - pis: (B, n) Mixture weights
    - mus: (B, n) Mixture means
    - sigmas: (B, n) Mixture standard deviations
    - var_values: (B,) Precomputed Value-at-Risk (VaR) for each sample

    Returns:
    - es_values: (B,) Expected Shortfall for each sample
    """
    # Compute z-scores for VaR
    z = (var_values[:, None] - mus) / sigmas  # Shape: (B, n)

    # Compute normal PDF and CDF for all components at once
    phi_z = norm.pdf(z)  # Shape: (B, n)
    Phi_z = norm.cdf(z)  # Shape: (B, n)

    # Compute numerator and denominator in vectorized form
    numerator = np.sum(pis * (mus * Phi_z - sigmas * phi_z), axis=1)  # (B,)
    denominator = np.sum(pis * Phi_z, axis=1)  # (B,)

    # Avoid division by zero
    denominator = np.where(denominator > 1e-12, denominator, 1.0)

    # Compute ES
    es_values = numerator / denominator

    return es_values


def calculate_es_for_quantiles(pis, mus, sigmas, quantiles):
    """
    Fully vectorized Expected Shortfall (ES) calculation.

    Parameters:
    - pis: (B, n) Mixture weights
    - mus: (B, n) Mixture means
    - sigmas: (B, n) Mixture standard deviations
    - quantiles: List of quantiles to compute ES for

    Returns:
    - es_results: (B, len(quantiles)) Expected Shortfall values
    """
    pis = np.asarray(pis)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    quantiles = np.asarray(quantiles)

    n_samples = pis.shape[0]
    n_q = len(quantiles)
    es_results = np.empty((n_samples, n_q))

    for j, q in enumerate(quantiles):
        print(f"Computing VaR for quantile {q}...")
        var_values = bracket_and_bisect(pis, mus, sigmas, q)
        print(f"Computing ES for quantile {q}...")
        es_results[:, j] = calculate_es_for_quantile(pis, mus, sigmas, var_values)

    return es_results
