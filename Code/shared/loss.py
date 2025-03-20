import numpy as np
import tensorflow as tf


def mdn_nll_numpy(num_mixtures):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate) using NumPy.
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_var].
    """

    def loss_fn(y_true, y_pred):
        # parse
        logits_pi = y_pred[:, :num_mixtures]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        pi = np.exp(
            logits_pi - np.logaddexp.reduce(logits_pi, axis=-1, keepdims=True)
        )  # softmax
        sigma = np.exp(0.5 * log_var)  # interpret as log-variance

        # expand dims for broadcast (B,) -> (B,1)
        y_true = np.expand_dims(y_true, axis=-1)

        # gaussian pdf for each mixture component
        normal_dist = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((y_true - mu) / sigma) ** 2
        )
        weighted_pdf = pi * normal_dist
        pdf_sum = np.sum(weighted_pdf, axis=1) + 1e-12  # avoid log(0)

        nll: np.ndarray = -np.log(pdf_sum)
        return nll

    return loss_fn


def mean_mdn_loss_numpy(num_mixtures):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate) using NumPy.
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_var].
    """
    unagged_loss_fn = mdn_nll_numpy(num_mixtures)

    def loss_fn(y_true, y_pred):
        nll = unagged_loss_fn(y_true, y_pred)
        return np.mean(nll)

    return loss_fn


def nll_loss_maf(model, X_test, y_test):
    """
    Calculate the Negative Log-Likelihood (NLL) for a non-parametric distribution produced by a Normalizing Flow model.

    Args:
        model: Trained LSTM-MAF model with `log_prob` method.
        X_test: Test input data (torch.Tensor).
        y_test: Actual observed returns (torch.Tensor).

    Returns:
        nll: Average Negative Log-Likelihood for the test set.
    """
    import torch

    model.eval()
    total_log_prob = 0
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for i in range(len(X_test)):
            specific_sample = X_test[i].unsqueeze(0)  # (1, lookback_days, num_features)
            actual_return = y_test[i].unsqueeze(0)  # (1,)

            # Compute log-probability of the actual return given the predicted distribution
            log_prob = model.log_prob(actual_return, specific_sample)
            total_log_prob += log_prob.item()

    # Return the average NLL
    nll = -total_log_prob / len(X_test)
    return nll


# def kde_negative_log_likelihood(y_obs, samples, bandwidth=0.1):
#     """
#     Compute -log p(y_obs) under a Gaussian KDE built from the given samples.

#     Args:
#       y_obs: scalar, the observed value
#       samples: array-like of shape (N,), the empirical distribution
#       bandwidth: float, the smoothing parameter for the Gaussian kernel

#     Returns:
#       A scalar representing the negative log-likelihood: -log p(y_obs)
#     """
#     samples = np.asarray(samples)
#     N = len(samples)

#     # Evaluate the kernel for each sample
#     # Gaussian kernel => 1/(sqrt(2*pi)*h) * exp( -(y-yi)^2 / (2h^2) )
#     # We'll do it in a vectorized way
#     coeff = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth)
#     diff_sq = (y_obs - samples)**2

#     kernel_vals = coeff * np.exp(-0.5 * diff_sq / (bandwidth**2))

#     # Average over the N kernels
#     pdf_est = np.mean(kernel_vals)

#     # If pdf_est is extremely small, the log might blow up
#     # We'll do a minimal floor to avoid -inf
#     pdf_est = max(pdf_est, 1e-40)

#     return -np.log(pdf_est)


def nll_loss_mean_and_log_var(y_true, means, log_vars):
    """
    Negative log-likelihood assuming a univariate Gaussian distribution given means
    and log variances.

    Args:
        y_true: target values (B, 1), i.e. actual values
        means: predicted means (B, 1)
        log_vars: predicted log variances (B, 1)

    Returns:
        nll: Negative log-likelihood (B,)
    """
    weights = np.ones_like(y_true)
    y_pred_combined = np.vstack([weights, means, log_vars]).T
    return mdn_nll_numpy(1)(y_true, y_pred_combined)


def nll_loss_mean_and_vol(y_true, means, vols):
    """
    Negative log-likelihood assuming a univariate Gaussian distribution given means
    and volatilities (standard deviations).

    Args:
        y_true: target values (B, 1), i.e. actual values
        means: predicted means (B, 1)
        vols: predicted volatilities (B, 1)

    Returns:
        nll: Negative log-likelihood (B,)
    """
    log_vars = 2 * np.log(vols)
    return nll_loss_mean_and_log_var(y_true, means, log_vars)


def mdn_nll_tf(num_mixtures, add_pi_penalty=False):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate).
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_var].
    """

    def loss_fn(y_true, y_pred):
        # parse
        logits_pi = y_pred[:, :num_mixtures]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        pi = tf.nn.softmax(logits_pi, axis=-1)
        sigma = tf.exp(0.5 * log_var)  # interpret as log-variance

        # expand dims for broadcast (B,) -> (B,1)
        y_true = tf.expand_dims(y_true, axis=-1)

        # gaussian pdf for each mixture component
        normal_dist = (1.0 / (sigma * tf.sqrt(2.0 * np.pi))) * tf.exp(
            -0.5 * ((y_true - mu) / sigma) ** 2
        )
        weighted_pdf = pi * normal_dist
        pdf_sum = tf.reduce_sum(weighted_pdf, axis=1) + 1e-12  # avoid log(0)

        nll = -tf.math.log(pdf_sum)
        if add_pi_penalty:
            # Add a penalty equal to the sum of the squares of the differences between
            # the pi values and 1/num_mixtures
            pi_penalty = tf.reduce_sum((pi - 1 / num_mixtures) ** 2, axis=1)
            nll += pi_penalty

        return tf.reduce_mean(nll)

    return loss_fn


import tensorflow as tf
import numpy as np


##############################################################################
# 1) Utility: Gauss–Legendre quadrature on [a,b]
##############################################################################
def gauss_legendre(npts, a, b):
    """
    Return xg, wg for Gauss-Legendre integration on [a,b], using
    NumPy's leggauss on [-1,1] and transforming to [a,b].
    """
    xg, wg = np.polynomial.legendre.leggauss(npts)  # on [-1, 1]
    xp = 0.5 * (b - a) * (xg + 1.0) + a
    wp = 0.5 * (b - a) * wg
    return xp.astype("float32"), wp.astype("float32")


##############################################################################
# 2) Normal CDF with broadcast
##############################################################################
def normal_cdf(x, mu, sigma):
    """
    Compute Φ((x - mu)/(√2 * sigma)) with shapes broadcast as needed.
    - x: [..., npts]
    - mu, sigma: the same leading dims, plus possibly a trailing dim of 1
    """
    sigma = tf.maximum(sigma, 1e-12)
    z = (x - mu) / (sigma * tf.sqrt(2.0))
    return 0.5 * (1.0 + tf.math.erf(z))


##############################################################################
# 3) The main routine: pairwise cdf L1 distance
#    Input:
#      mu, sigma: shape [batch, M]
#      xg, wg: shape [npts]
#    Output shape: [batch, M, M]
##############################################################################
def l1_cdf_distance(mu, sigma, xg, wg):
    """
    Numerically approximate ∫|Φ_k(x) - Φ_j(x)| dx for each pair (k,j).
    Return shape [batch, M, M].
    """
    # 1) Expand mu, sigma from [B, M] to [B, M, 1, 1] and [B, 1, M, 1]
    #    so we get a final shape of [B, M, M, npts].
    B = tf.shape(mu)[0]
    M = tf.shape(mu)[1]

    # Reshape each to 4D
    mu1 = tf.reshape(mu, [B, M, 1, 1])  # shape (B,M,1,1)
    mu2 = tf.reshape(mu, [B, 1, M, 1])  # shape (B,1,M,1)
    s1 = tf.reshape(sigma, [B, M, 1, 1])
    s2 = tf.reshape(sigma, [B, 1, M, 1])

    # 2) Expand xg, wg to shape [1,1,1,npts]
    x_expanded = tf.reshape(xg, [1, 1, 1, -1])  # => (1,1,1,npts)
    w_expanded = tf.reshape(wg, [1, 1, 1, -1])  # => (1,1,1,npts)

    # 3) Evaluate CDF for each pair k,j over the grid
    cdf1 = normal_cdf(x_expanded, mu1, s1)  # => shape [B,M,M,npts]
    cdf2 = normal_cdf(x_expanded, mu2, s2)

    # 4) Integrate the absolute difference
    diff = tf.abs(cdf1 - cdf2)  # [B,M,M,npts]
    integrand = diff * w_expanded  # multiply by Gauss-Legendre weights

    # 5) Sum over npts
    return tf.reduce_sum(integrand, axis=-1)  # => shape [B,M,M]


##############################################################################
# 4) Single-component CRPS formula for Normal(μ,σ)
##############################################################################
def crps_normal(y, mu, sigma):
    """
    CRPS for a single Normal(μ,σ) vs. scalar y:
      CRPS(N(μ,σ), y) = σ * [ z(2Φ(z) - 1) + 2φ(z) - 1/√π ],
      where z = (y - μ)/(σ √2).
    Shapes broadcast as [B, M] if y is [B,1].
    """
    sigma = tf.maximum(sigma, 1e-12)
    z = (y - mu) / (sigma * tf.sqrt(2.0))

    pdf_z = 1.0 / tf.sqrt(2.0 * np.pi) * tf.exp(-0.5 * z * z)
    cdf_z = 0.5 * (1.0 + tf.math.erf(z))

    return sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / tf.sqrt(np.pi))


##############################################################################
# 5) Final MDN-CRPS loss
##############################################################################
def mdn_crps_tf(
    num_mixtures,
    add_pi_penalty=False,
    add_mu_penalty=False,
    add_sigma_penalty=False,
    npts=16,
    tmin=-0.08,
    tmax=0.08,
):
    """
    Mixture of Gaussians CRPS:
      y_pred -> [batch, 3*num_mixtures], with
        logits_pi = y_pred[:, :num_mixtures]
        mu        = y_pred[:, num_mixtures:2*num_mixtures]
        log_var   = y_pred[:, 2*num_mixtures:]
    """
    # Precompute Gauss–Legendre points and weights just once
    x_np, w_np = gauss_legendre(npts, tmin, tmax)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    w_tf = tf.constant(w_np, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # 1) Parse mixture parameters
        logits_pi = y_pred[:, :num_mixtures]  # [B, M]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        pi = tf.nn.softmax(logits_pi, axis=-1)  # [B, M]
        sigma = tf.exp(0.5 * log_var)  # [B, M]

        # 2) Single mixture CRPS: sum_k pi_k * CRPS(N_k, y)
        y_expanded = tf.expand_dims(y_true, axis=-1)  # [B,1]
        crps_k = crps_normal(y_expanded, mu, sigma)  # [B, M]
        term_single = tf.reduce_sum(pi * crps_k, axis=-1)  # [B]

        # 3) Pairwise mixture part: 0.5 * sum_{k,j} pi_k pi_j * ∫|F_k - F_j|
        l1_dist = l1_cdf_distance(mu, sigma, x_tf, w_tf)  # [B, M, M]
        # broadcast pi -> [B, M, 1] and [B, 1, M]
        pi_k = tf.expand_dims(pi, axis=2)  # [B, M, 1]
        pi_j = tf.expand_dims(pi, axis=1)  # [B, 1, M]
        term_pairwise = 0.5 * tf.reduce_sum(pi_k * pi_j * l1_dist, axis=[1, 2])

        crps_val = term_single - term_pairwise

        # 4) Optional penalties
        if add_pi_penalty:
            penalty = tf.reduce_sum((pi - 1.0 / num_mixtures) ** 2, axis=-1)
            crps_val += penalty

        if add_mu_penalty:
            penalty = tf.reduce_sum(mu**2, axis=-1)
            crps_val += penalty

        if add_sigma_penalty:
            penalty = tf.reduce_sum(sigma**2, axis=-1)
            crps_val += penalty

        return crps_val

    return loss_fn


def crps_normal_univariate(y_true, mus, sigmas, npts=16, tmin=-0.08, tmax=0.08):
    """
    Compute the CRPS for a univariate Normal distribution by leveraging
    the MDN-CRPS implementation with a single mixture component.

    Parameters:
      y_true: array-like, shape [B], true scalar values.
      mus:    array-like, shape [B], means.
      sigmas: array-like, shape [B], standard deviations.
      npts, tmin, tmax: Parameters for Gauss-Legendre quadrature (see mdn_crps_tf).

    Returns:
      Tensor of shape [B] with the CRPS for each sample.
    """
    # Convert inputs to tf.float32 tensors.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    mus = tf.convert_to_tensor(mus, dtype=tf.float32)
    sigmas = tf.convert_to_tensor(sigmas, dtype=tf.float32)

    B = tf.shape(mus)[0]
    logits_pi = tf.zeros([B, 1], dtype=tf.float32)  # shape [B,1]
    mu = tf.reshape(mus, [-1, 1])  # shape [B,1]
    log_var = 2.0 * tf.math.log(sigmas)  # so that exp(0.5*log_var)=sigmas
    log_var = tf.reshape(log_var, [-1, 1])  # shape [B,1]

    # Construct y_pred as required: [logits_pi, mu, log_var] => shape [B, 3]
    y_pred = tf.concat([logits_pi, mu, log_var], axis=1)

    # Create the loss function for a single-mixture MDN CRPS
    loss_fn = mdn_crps_tf(num_mixtures=1, npts=npts, tmin=tmin, tmax=tmax)

    # Compute and return the CRPS value; note that for num_mixtures=1 the
    # pairwise term vanishes and we recover crps_normal.
    return loss_fn(y_true, y_pred)


def mean_mdn_crps_tf(
    num_mixtures,
    add_pi_penalty=False,
    add_mu_penalty=False,
    add_sigma_penalty=False,
    npts=16,
    tmin=-0.08,
    tmax=0.08,
):
    """
    Mixture of Gaussians CRPS:
      y_pred -> [batch, 3*num_mixtures], with
        logits_pi = y_pred[:, :num_mixtures]
        mu        = y_pred[:, num_mixtures:2*num_mixtures]
        log_var   = y_pred[:, 2*num_mixtures:]
    """
    unagged_loss_fn = mdn_crps_tf(
        num_mixtures,
        add_pi_penalty,
        add_mu_penalty,
        add_sigma_penalty,
        npts,
        tmin,
        tmax,
    )

    def loss_fn(y_true, y_pred):
        crps_val = unagged_loss_fn(y_true, y_pred)
        return tf.reduce_mean(crps_val)

    return loss_fn


def fz_loss(returns: np.ndarray, VaR: np.ndarray, ES: np.ndarray, quantile: float):
    """
    Calculates the FZ loss, as specified by Patton et al. (2019).

    Args:
        returns: Array of returns.
        VaR: Value at Risk estimates for the quantile level (negative number = loss).
        ES: Expected Shortfall estimates for the quantile level (negative number = loss).
        quantile: Quantile level (lower quantile => bigger loss). **NB**: Not confidence level.
    """
    L = (returns < VaR).astype(int)
    term1 = -L * (VaR - returns) / (quantile * ES)
    term2 = VaR / ES
    term3 = np.log(-ES)
    return term1 + term2 + term3 - 1


def al_loss(returns: np.ndarray, VaR: np.ndarray, ES: np.ndarray, quantile: float):
    """
    Calculates Assymetric Laplace Density log score, as introduced by Taylor (2017).

    Args:
        returns: Array of returns.
        VaR: Value at Risk estimates for the quantile level (negative number = loss).
        ES: Expected Shortfall estimates for the quantile level (negative number = loss).
        quantile: Quantile level. **NB**: Not confidence level.
    """
    L = (returns < VaR).astype(int)
    term1 = -np.log((quantile - 1) / ES)
    term2 = -(returns - VaR) * (quantile - L) / (quantile * ES)
    return term1 + term2
