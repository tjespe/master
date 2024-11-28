import tensorflow as tf


# Define function for making predictions with MC Dropout
@tf.function
def predict_with_mc_dropout_tf(model, X, T=100, mean_i=0, var_i=1):
    preds = []
    for i in range(T):
        y_p = model(X, training=True)  # Keep dropout active
        preds.append(y_p)

    # Stack predictions into a tensor
    preds = tf.stack(preds)

    # Calculate statistics using TensorFlow operations
    if mean_i is not None:
        expected_returns = tf.reduce_mean(preds[:, :, mean_i], axis=0)
        epistemic_uncertainty_expected_returns = tf.math.reduce_std(
            preds[:, :, mean_i], axis=0
        )
    else:
        expected_returns = None
        epistemic_uncertainty_expected_returns = None

    if var_i is not None:
        log_variance_preds = preds[:, :, var_i]
        variance_estimates = tf.math.exp(log_variance_preds)
        variance_estimates = tf.maximum(variance_estimates, 1e-6)
        volatility_estimates = tf.math.sqrt(variance_estimates)
        volatility_estimate_per_day = tf.reduce_mean(volatility_estimates, axis=0)
        epistemic_uncertainty_volatility_estimates = tf.math.reduce_std(
            volatility_estimates, axis=0
        )
    else:
        volatility_estimate_per_day = None
        epistemic_uncertainty_volatility_estimates = None
        variance_estimates = None

    return {
        "expected_returns": expected_returns,
        "epistemic_uncertainty_expected_returns": epistemic_uncertainty_expected_returns,
        "volatility_estimates": volatility_estimate_per_day,
        "epistemic_uncertainty_volatility_estimates": epistemic_uncertainty_volatility_estimates,
        "preds": preds,
    }


def predict_with_mc_dropout(model, X, T=100, mean_i=0, var_i=1):
    # Call the @tf.function-decorated function
    results = predict_with_mc_dropout_tf(model, X, T, mean_i, var_i)

    # Convert TensorFlow tensors to NumPy arrays
    return {key: value.numpy() for key, value in results.items() if value is not None}
