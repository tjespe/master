import numpy as np
import tensorflow as tf


def mdn_loss_numpy(num_mixtures):
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

        nll = -np.log(pdf_sum)
        return nll

    return loss_fn


def mean_mdn_loss_numpy(num_mixtures):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate) using NumPy.
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_var].
    """
    unagged_loss_fn = mdn_loss_numpy(num_mixtures)

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


def nll_loss_mean_and_log_var(y_true, means, log_vars):
    """
    Negative log-likelihood assuming a univariate Gaussian distribution given means
    and log variances.

    Args:
        y_true: target values (B, 1), i.e. actual values
        means: predicted means (B, 1)
        log_vars: predicted log variances (B, 1)
    """
    weights = np.ones_like(y_true)
    y_pred_combined = np.vstack([weights, means, log_vars]).T
    return mean_mdn_loss_numpy(1)(y_true, y_pred_combined)


def nll_loss_mean_and_vol(y_true, means, vols):
    """
    Negative log-likelihood assuming a univariate Gaussian distribution given means
    and volatilities (standard deviations).

    Args:
        y_true: target values (B, 1), i.e. actual values
        means: predicted means (B, 1)
        vols: predicted volatilities (B, 1)
    """
    log_vars = 2 * np.log(vols)
    return nll_loss_mean_and_log_var(y_true, means, log_vars)


def mdn_loss_tf(num_mixtures, add_pi_penalty=False):
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


################################################################################
# 1) Gauss-Legendre utility
################################################################################
def gauss_legendre(npts, a, b):
    """
    Returns the 1D Gauss-Legendre sample points (x) and weights (w)
    for integrating on [a, b] with npts nodes.
    We use np.polynomial.legendre.leggauss and transform from [-1,1] to [a,b].
    """
    # xg, wg for the standard interval [-1, 1]
    xg, wg = np.polynomial.legendre.leggauss(npts)
    # Transform to [a, b]
    # x in [a,b], weight = (b-a)/2 * wg
    xp = 0.5 * (b - a) * (xg + 1.0) + a
    wp = 0.5 * (b - a) * wg
    return xp.astype(np.float32), wp.astype(np.float32)


################################################################################
# 2) CRPS for a single Normal(μ,σ) via known closed form
#    CRPS(N(μ,σ), y) = σ * [ z(2Φ(z) - 1) + 2φ(z) - 1/√π ],
#    where z = (y - μ) / (√2 σ).
################################################################################
def crps_normal(y, mu, sigma):
    sigma = tf.maximum(sigma, 1e-12)
    z = (y - mu) / (sigma * tf.sqrt(2.0))

    pdf_z = 1.0 / tf.sqrt(2.0 * np.pi) * tf.exp(-0.5 * z * z)
    cdf_z = 0.5 * (1.0 + tf.math.erf(z))

    return sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / tf.sqrt(np.pi))


################################################################################
# 3) Numeric approximation of the L1 distance ∫ |Φk - Φj|.
#    We'll do Gauss-Legendre quadrature over [tmin, tmax].
################################################################################
def normal_cdf(x, mu, sigma):
    z = (x - mu) / (tf.maximum(sigma, 1e-12) * tf.sqrt(2.0))
    return 0.5 * (1.0 + tf.math.erf(z))


def l1_cdf_distance(mu1, sig1, mu2, sig2, xg, wg):
    """
    mu1, sig1, mu2, sig2: shape [batch, mix, mix]
    xg, wg: Gauss-Legendre points and weights, shape [npts]
    Returns ∫ |F1(x) - F2(x)| dx as [batch, mix, mix].
    """
    # Expand xg to broadcast: shape (1,1,1,npts)
    x_expanded = tf.reshape(xg, [1, 1, 1, -1])

    # Evaluate cdfs at all pairs in one shot
    cdf1 = normal_cdf(x_expanded, mu1, sig1)  # [batch, mix, mix, npts]
    cdf2 = normal_cdf(x_expanded, mu2, sig2)

    diff = tf.abs(cdf1 - cdf2)  # [batch, mix, mix, npts]

    # Multiply by weights, sum over npts
    # shape → [batch, mix, mix]
    return tf.reduce_sum(diff * wg, axis=-1)


################################################################################
# 4) The final CRPS mixture loss:
#      CRPS( ∑ πᵏ N(μᵏ,σᵏ) ) =
#         ∑ πᵏ CRPS(Nᵏ, y)  -  0.5 ∑ₖ∑ⱼ πᵏπⱼ ∫ |Fᵏ - Fⱼ|.
#    The second term is just a mixture-pair average of cdf differences.
################################################################################
def mdn_crps_tf(num_mixtures, add_pi_penalty=False, npts=16, tmin=-0.08, tmax=0.08):
    """
    Returns a CRPS-based loss for univariate Gaussian mixture outputs.
    y_pred is shape [batch, 3 * num_mixtures], split into:
      logits_pi = y_pred[:, :num_mixtures]
      mu        = y_pred[:, num_mixtures:2*num_mixtures]
      log_var   = y_pred[:, 2*num_mixtures:]
    """
    # Precompute Gauss-Legendre points & weights once
    xg_np, wg_np = gauss_legendre(npts, tmin, tmax)
    # Convert to TF constants
    xg_tf = tf.constant(xg_np, dtype=tf.float32)  # [npts]
    wg_tf = tf.constant(wg_np, dtype=tf.float32)  # [npts]

    def loss_fn(y_true, y_pred):
        # Parse mixture parameters
        logits_pi = y_pred[:, :num_mixtures]  # [batch, mix]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        pi = tf.nn.softmax(logits_pi, axis=-1)  # [batch, mix]
        sigma = tf.exp(0.5 * log_var)  # [batch, mix]

        y = tf.expand_dims(y_true, axis=-1)  # [batch, 1]

        # 1) Sum of single mixture CRPS
        crps_single = crps_normal(y, mu, sigma)  # [batch, mix]
        term1 = tf.reduce_sum(pi * crps_single, axis=-1)  # [batch]

        # 2) Pairwise cdf overlaps
        # Expand to shape [batch, mix, mix] for mu, sigma, pi
        mu1 = tf.expand_dims(mu, 2)  # [batch, mix, 1]
        mu2 = tf.expand_dims(mu, 1)  # [batch, 1, mix]
        sig1 = tf.expand_dims(sigma, 2)  # [batch, mix, 1]
        sig2 = tf.expand_dims(sigma, 1)  # [batch, 1, mix]
        pi1 = tf.expand_dims(pi, 2)  # [batch, mix, 1]
        pi2 = tf.expand_dims(pi, 1)  # [batch, 1, mix]

        l1_dist = l1_cdf_distance(mu1, sig1, mu2, sig2, xg_tf, wg_tf)
        # Weighted sum
        term2 = 0.5 * tf.reduce_sum(pi1 * pi2 * l1_dist, axis=[1, 2])

        # Combine: CRPS = term1 - term2
        crps_val = term1 - term2

        # Optional penalty to keep pi near uniform
        if add_pi_penalty:
            penalty = tf.reduce_sum((pi - 1.0 / num_mixtures) ** 2, axis=-1)
            crps_val += penalty

        return tf.reduce_mean(crps_val)

    return loss_fn
