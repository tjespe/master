import numpy as np
import tensorflow as tf
from scipy.special import gammaln
from scipy.stats import t
from scipy.integrate import quad
from arch.univariate.distribution import SkewStudent

from shared.skew_t import rvs_skewt


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


def student_t_nll(y_true, means, vols, nu):
    """
    Negative log-likelihood for a Student-t distribution.

    Args:
        y_true: (B, 1) actual values
        means: (B, 1) predicted means
        vols: (B, 1) predicted standard deviations (sigma)
        nu: scalar or (B, 1), degrees of freedom parameter (ν)

    Returns:
        nll: Negative log-likelihood (B,)
    """
    y_true = np.squeeze(y_true)
    means = np.squeeze(means)
    vols = np.squeeze(vols)

    # Ensure nu is an array of same shape, if it's scalar
    nu = np.full_like(y_true, nu) if np.isscalar(nu) else np.squeeze(nu)

    # Compute standardized residuals: (y - mu) / sigma
    standardized_residuals = (y_true - means) / vols

    # Compute the gamma-related term: log(Gamma((nu+1)/2)) - log(Gamma(nu/2))
    log_gamma_term = gammaln((nu + 1) / 2) - gammaln(nu / 2)

    # The coefficient part in the log PDF:
    # -0.5 * log(nu*pi) - log(sigma)
    log_coeff = -0.5 * np.log(nu * np.pi) - np.log(vols)

    # The kernel part: -(nu+1)/2 * log(1 + (1/nu) * standardized_residuals^2)
    log_kernel = -(nu + 1) / 2 * np.log(1 + (1 / nu) * standardized_residuals**2)

    # Log probability density for each observation
    log_prob = log_gamma_term + log_coeff + log_kernel

    # Return the negative log-likelihood
    nll = -log_prob
    return nll


def skewt_nll(y_true, means, vols, nu, skew):
    """
    Compute the negative log-likelihood (NLL) for a skewed t-distribution
    forecast (Hansen skew-t). This function applies the change-of-variable
    correction when moving from the standardized residuals to the original scale.

    Parameters:
    -----------
    y_true : array-like, shape (B,)
        The observed values.
    means : array-like, shape (B,)
        The forecasted means.
    vols : array-like, shape (B,)
        The forecasted volatility (scale parameter).
    nu : array-like or scalar
        Degrees-of-freedom parameter.
    skew : array-like or scalar
        Skewness parameter.

    Returns:
    --------
    nll : array-like, shape (B,)
        The negative log-likelihood for each observation.
    """
    # Ensure inputs are numpy arrays of appropriate shape
    y_true = np.squeeze(np.array(y_true))
    means = np.squeeze(np.array(means))
    vols = np.squeeze(np.array(vols))

    # If nu or skew are scalar, promote them to arrays of appropriate shape.
    if np.isscalar(nu):
        nu = np.full_like(y_true, nu)
    else:
        nu = np.squeeze(np.array(nu))
    if np.isscalar(skew):
        skew = np.full_like(y_true, skew)
    else:
        skew = np.squeeze(np.array(skew))

    # Standardize the residuals.
    z = (y_true - means) / vols

    # Initialize the skewed Student-t distribution object.
    skewt = SkewStudent()

    # Evaluate the log pdf for the standardized residuals using the skew-t.
    # The pdf is computed on the standardized scale, so we subtract log(vols)
    # to account for the change-of-variable from x to z = (x - mu) / sigma.
    log_pdf = skewt.logpdf(z, nu=nu, lam=skew) - np.log(vols)

    # The negative log-likelihood for each observation.
    nll = -log_pdf
    return nll


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
    tmin=-0.5,
    tmax=0.5,
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


def crps_normal_univariate(y_true, mus, sigmas, npts=16, tmin=-0.5, tmax=0.5):
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
    tmin=-0.5,
    tmax=0.5,
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


def crps_student_t(x, mu, sigma, nu):
    """
    Compute the CRPS for a Student-t distribution with location mu, scale sigma, and degrees of freedom nu at observation x.

    Args:
        x (float): The observation.
        mu (float): Location parameter of the Student-t distribution.
        sigma (float): Scale parameter.
        nu (float): Degrees of freedom.

    Returns:
        float: CRPS value.
    """

    # Define the Student-t CDF with parameters mu, sigma, nu.
    def F(y):
        return t.cdf((y - mu) / sigma, df=nu)

    # Define the integrand of the CRPS integral.
    def integrand(y):
        indicator = 1.0 if y >= x else 0.0
        return (F(y) - indicator) ** 2

    # Perform numerical integration over the real line.
    crps_value, _ = quad(integrand, -np.inf, np.inf)
    return crps_value


def crps_skewt(x, mu, sigma, nu, lam, nsim=1000, random_state=None):
    """
    Monte‐Carlo CRPS for a location‐scale skew‐t, returns an array of per‐obs CRPS.

    x     : array‐like, shape (B,)  observed values
    mu    : array‐like, shape (B,)  forecast means
    sigma : array‐like, shape (B,)  forecast scales (σ)
    nu    : array‐like or scalar     dfs η
    lam   : array‐like or scalar     skewness λ
    nsim  : int                      number of MC draws per obs
    random_state : int or None      seed for reproducibility
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    nu = np.asarray(nu, dtype=float)
    lam = np.asarray(lam, dtype=float)

    B = x.shape[0]
    # broadcast scalars
    if nu.size == 1:
        nu = np.full(B, nu.item(), dtype=float)
    if lam.size == 1:
        lam = np.full(B, lam.item(), dtype=float)

    crps_vals = np.empty(B, dtype=float)

    # use one RNG if you like reproducibility
    # but since rvs_skewt takes its own random_state, we'll pass None
    for i in range(B):
        z_obs = (x[i] - mu[i]) / sigma[i]
        # now nu[i] and lam[i] are scalars
        z_samples = rvs_skewt(nsim, nu=nu[i], lam=lam[i])

        term1 = np.mean(np.abs(z_samples - z_obs))
        diffs = np.abs(z_samples[:, None] - z_samples[None, :])
        # exclude the zero‐diagonal
        mask = ~np.eye(nsim, dtype=bool)
        term2 = 0.5 * diffs[mask].mean()

        crps_vals[i] = sigma[i] * (term1 - term2)

    return crps_vals


##############################################################################
# ECE for Mixture Density Networks
##############################################################################
def ece_mdn(num_mixtures, y_true, y_pred, n_bins=20):
    """
    Compute an approximate Expected Calibration Error (ECE) for a univariate
    Mixture of Gaussians. We:
      1) Parse the mixture parameters: [logits_pi, mu, log_var].
      2) For each data point i, compute CDF_i = sum_k pi_k * Phi((y_i - mu_k)/sigma_k).
      3) Bin the CDF values and measure how far the empirical coverage deviates
         from the nominal coverage, then average.

    Args:
      num_mixtures: int, number of mixture components
      y_true: shape (B,)
      y_pred: shape (B, 3*num_mixtures)
      n_bins: number of calibration bins

    Returns:
      ece_value: scalar float
    """
    # 1) Parse mixture parameters
    logits_pi = y_pred[:, :num_mixtures]  # (B, M)
    mu = y_pred[:, num_mixtures : 2 * num_mixtures]  # (B, M)
    log_var = y_pred[:, 2 * num_mixtures :]  # (B, M)

    # softmax for pi
    pi = np.exp(
        logits_pi - np.logaddexp.reduce(logits_pi, axis=-1, keepdims=True)
    )  # (B, M)
    sigma = np.exp(0.5 * log_var)  # interpret log_var as log-variance => (B, M)

    # 2) Compute CDF_i for each data point i
    #    For mixture k:  cdf_k_i = Phi((y_i - mu_k)/sigma_k).
    #    Then total CDF_i = sum_k pi_k * cdf_k_i.
    y_true_expanded = y_true[:, None]  # (B, 1)
    z = (y_true_expanded - mu) / (sigma + 1e-12)  # shape (B, M)
    cdf_components = normal_cdf(z, 0.0, 1.0)  # shape (B, M)
    cdf_vals = np.sum(pi * cdf_components, axis=1)  # shape (B,)

    # 3) Bin-based ECE
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins  # e.g. 0.05, 0.15, ..., 0.95
    ece = 0.0
    for q in bin_centers:
        p_emp = np.mean(cdf_vals <= q)
        ece += abs(p_emp - q)
    ece /= n_bins

    return ece


##############################################################################
# ECE for Univariate Gaussians (calls the MDN version with one mixture)
##############################################################################
def ece_gaussian(y_true, means, log_vars, n_bins=20):
    """
    Compute ECE for a univariate Gaussian by wrapping it into the MDN structure
    with a single mixture component.
    """
    B = len(y_true)
    # We'll build y_pred = [logits_pi=0, mu=means, log_var=log_vars]
    # => shape (B, 3) for one mixture
    # For logits_pi=0 => pi=1
    logits_pi = np.zeros((B, 1), dtype=np.float32)
    means = means.reshape(B, 1)
    log_vars = log_vars.reshape(B, 1)
    y_pred = np.concatenate([logits_pi, means, log_vars], axis=1)

    return ece_mdn(num_mixtures=1, y_true=y_true, y_pred=y_pred, n_bins=n_bins)


##############################################################################
# ECE for Student-t Distribution
##############################################################################
def ece_student_t(y_true, means, vols, nu, n_bins=20):
    """
    Compute an approximate ECE for a univariate Student-t distribution with
    location=means, scale=vols, and degrees of freedom=nu.  We do a binning
    of the distribution CDF at the observed data.

    Args:
      y_true: shape (B,)
      means: shape (B,) or broadcastable
      vols: shape (B,) or broadcastable
      nu: scalar or shape (B,) degrees of freedom
      n_bins: int, number of bins
    """
    from scipy.stats import t

    y_true = np.squeeze(y_true)
    means = np.squeeze(means)
    vols = np.squeeze(vols)

    # If nu is scalar, make an array
    if np.isscalar(nu):
        nu = np.full_like(y_true, nu)

    # 1) Compute cdf_i = T.cdf( (y_i - mu_i)/sigma_i; df=nu_i )
    #    We'll rely on scipy.stats.t
    standardized = (y_true - means) / (vols + 1e-12)
    cdf_vals = t.cdf(standardized, df=nu)  # shape (B,)

    # 2) Bin-based ECE
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    ece = 0.0
    for q in bin_centers:
        p_emp = np.mean(cdf_vals <= q)
        ece += abs(p_emp - q)
    ece /= n_bins

    return ece


def ece_skewt(y_true, means, vols, nus, lams, n_bins=20):
    """
    Exact ECE for a univariate skewed‐t forecast via arch’s CDF.

    y_true : (B,) observations
    means  : (B,) location forecasts
    vols   : (B,) scale forecasts (σ)
    nus    : scalar or (B,) degrees of freedom η
    lams   : scalar or (B,) skewness λ in (–1,1)
    n_bins : int    number of probability bins
    """
    y = np.asarray(y_true).ravel()
    μ = np.asarray(means).ravel()
    σ = np.asarray(vols).ravel()
    B = len(y)

    # expand scalars
    if np.isscalar(nus):
        η = np.full(B, float(nus))
    else:
        η = np.asarray(nus).ravel()
    if np.isscalar(lams):
        λ = np.full(B, float(lams))
    else:
        λ = np.asarray(lams).ravel()

    # standardize residuals
    z = (y - μ) / σ

    dist = SkewStudent()
    cdf_vals = np.empty(B, dtype=float)

    # call the CDF correctly: cdf(resids, parameters=[eta, lam])
    for i in range(B):
        # dist.cdf expects a sequence of resids and a fixed parameter vector
        cdf_vals[i] = dist.cdf([z[i]], parameters=[η[i], λ[i]])[0]

    # now bin‐based ECE
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    ece = 0.0
    for q in bin_centers:
        p_emp = np.mean(cdf_vals <= q)
        ece += abs(p_emp - q)
    return ece / n_bins
