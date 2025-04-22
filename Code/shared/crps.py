import numpy as np

##################################
#    CRPS for mixture model      #
##################################


def sample_from_mixture(pi, mu, sigma, size=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K = len(pi)
    mixture_choices = rng.choice(K, size=size, p=pi)
    samples = rng.normal(loc=mu[mixture_choices], scale=sigma[mixture_choices])
    return samples


def crps_sample_based(y_true, forecast_samples):
    forecast_samples = np.asarray(forecast_samples)
    N = len(forecast_samples)
    term1 = np.mean(np.abs(forecast_samples - y_true))
    term2 = 0.5 * np.mean(
        np.abs(forecast_samples.reshape(-1, 1) - forecast_samples.reshape(1, -1))
    )
    return term1 - term2


def crps_mdn_numpy(num_mixtures, num_samples=1000, seed=None):
    rng = np.random.default_rng(seed)

    def crps_fn(y_true, y_pred):
        # y_true: (B,)
        # y_pred: (B, 3 * num_mixtures) => parse [logits_pi, mu, log_var]
        B = y_true.shape[0]
        crps_values = np.zeros(B)

        logits_pi = y_pred[:, :num_mixtures]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        # Convert to mixing weights (pi) and std dev (sigma)
        pi = np.exp(logits_pi - np.logaddexp.reduce(logits_pi, axis=1, keepdims=True))
        sigma = np.exp(0.5 * log_var)

        for i in range(B):
            # Draw samples from the mixture for row i
            s = sample_from_mixture(pi[i], mu[i], sigma[i], size=num_samples, rng=rng)
            crps_values[i] = crps_sample_based(y_true[i], s)

        return crps_values

    return crps_fn


#####################################################
# Wrap it for a single univariate Gaussian (K = 1). #
#####################################################


def crps_loss_mean_and_log_var(y_true, means, log_vars, num_samples=1000, seed=None):
    """
    Compute sample-based CRPS for a univariate Gaussian.
    We treat it as a single-mixture MDN (K=1).

    Args:
        y_true:  (B,) array of actual values
        means:   (B,) array of predicted means
        log_vars:(B,) array of predicted log-variances
    """
    B = len(y_true)

    # For single mixture, shape is (B, 3):
    #   [logits_pi, mu, log_var]
    #   logits_pi can be 0 => softmax([0]) = [1.0].
    logits_pi = np.zeros((B, 1))
    mu = means.reshape(B, 1)
    log_var = log_vars.reshape(B, 1)

    # Combine into y_pred
    y_pred = np.hstack([logits_pi, mu, log_var])

    # Instantiate the CRPS function for a single mixture
    crps_fn = crps_mdn_numpy(num_mixtures=1, num_samples=num_samples, seed=seed)

    # Return the batch mean CRPS
    return crps_fn(y_true, y_pred)


def crps_loss_mean_and_vol(y_true, means, vols, num_samples=1000, seed=None):
    """
    Compute sample-based CRPS for a univariate Gaussian.
    We treat it as a single-mixture MDN (K=1).

    Args:
        y_true:  (B,) array of actual values
        means:   (B,) array of predicted means
        vols:    (B,) array of predicted volatilities (std. deviations)
    """
    vars = vols**2
    log_vars = np.log(vars)
    return crps_loss_mean_and_log_var(y_true, means, log_vars, num_samples, seed)
