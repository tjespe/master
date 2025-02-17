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
    return mdn_loss_numpy(1)(y_true, y_pred_combined)


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


def mdn_loss_tf(num_mixtures):
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
        return tf.reduce_mean(nll)

    return loss_fn
