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
