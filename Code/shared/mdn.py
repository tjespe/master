import numpy as np
import tensorflow as tf

from scipy.optimize import brentq
from scipy.stats import norm


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
    parse out pi, mu, sigma. Returns (pi, mu, sigma) each shape = (batch, n_mixtures).
    """
    logits_pi = mdn_out[:, :n_mixtures]
    mu = mdn_out[:, n_mixtures : 2 * n_mixtures]
    log_var = mdn_out[:, 2 * n_mixtures :]

    pi = tf.nn.softmax(logits_pi, axis=-1)
    sigma = tf.exp(0.5 * log_var)

    return pi, mu, sigma


def univariate_mixture_mean_and_var_approx(pi, mu, sigma):
    """
    Creates a univariate approximation of the mixture distribution.

    For univariate mixture:
      mixture_mean = sum(pi_k * mu_k)
      mixture_var  = sum(pi_k * (sigma_k^2 + mu_k^2)) - mixture_mean^2
    """
    mixture_mean = tf.reduce_sum(pi * mu, axis=1)
    mixture_mean_sq = tf.square(mixture_mean)
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
        mean_s, var_s = univariate_mixture_mean_and_var(pi, mu, sigma)
        mc_means.append(mean_s.numpy())
        mc_vars.append(var_s.numpy())

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
