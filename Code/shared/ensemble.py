import tensorflow as tf

from shared.mdn import parse_mdn_output


class MDNEnsemble(tf.keras.Model):
    """
    A univariate MDN ensemble that:
      1) Concatenates each submodel's n_mixtures into one large mixture,
      2) Computes epistemic uncertainty = var(across-submodel-means).
    """

    def __init__(self, submodels, n_mixtures, name="mdn_ensemble"):
        super().__init__(name=name)
        # submodels: list of trained tf.keras.Models
        #   each must output shape (batch_size, 3*n_mixtures)
        self.submodels = submodels
        self.n_mixtures = n_mixtures

    def call(self, inputs, training=False):
        # Each submodel's mixture weights get scaled by 1/ensemble_size
        # then concatenated along axis=-1.
        pi_list = []
        mu_list = []
        logvar_list = []  # because each submodel originally outputs log_var

        for m in self.submodels:
            output = m(inputs, training=training)
            pi, mu, sigma = parse_mdn_output(output, self.n_mixtures)

            # Scale mixture weights so final sum is 1
            pi = pi / len(self.submodels)

            # Convert sigma -> log_var for final MDN output
            # sigma_i = exp(0.5 * log_var_i) => log_var_i = 2 * log(sigma_i)
            log_var = 2.0 * tf.math.log(sigma + 1e-8)

            pi_list.append(pi)
            mu_list.append(mu)
            logvar_list.append(log_var)

        pi_ens = tf.concat(pi_list, axis=-1)  # (batch_size, n_models*n_mixtures)
        mu_ens = tf.concat(mu_list, axis=-1)
        log_var_ens = tf.concat(logvar_list, axis=-1)

        # Convert pi_ens -> logits for final MDN
        logits_pi_ens = tf.math.log(pi_ens + 1e-8)

        # 4) Create final MDN output shape = (batch_size, 3*n_models*n_mixtures)
        ensemble_mdn_output = tf.concat([logits_pi_ens, mu_ens, log_var_ens], axis=-1)

        # 5) Compute epistemic variance = variance of submodel means
        # submodel i's mixture mean => sum_j pi[i,j]*mu[i,j], shape: (batch_size,)
        mixture_means = tf.reduce_sum(
            pi * mu, axis=-1
        )  # shape => (ensemble_size, batch_size)
        # variance across ensemble_size dimension => shape: (batch_size,)
        epistemic_var = tf.math.reduce_variance(mixture_means, axis=0)

        return ensemble_mdn_output, epistemic_var

    def get_config(self):
        """
        Needed to allow self.save(...) / load_model(...).
        Each submodel also needs to be serializable if you want truly portable saves.
        """
        config = super().get_config().copy()
        config.update(
            {
                "n_mixtures": self.n_mixtures,
                # Potentially store submodels by config, but for real usage each submodel
                # also needs a working get_config / from_config or you must handle them manually.
                "submodels": [m.get_config() for m in self.submodels],
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # You'd have to rebuild submodels from their configs here.
        # In practice, for a full reconstruction, you typically rely on
        # SavedModel format or handle submodels explicitly.
        n_mixtures = config["n_mixtures"]
        # For demonstration only:
        submodels_placeholder = []
        return cls(submodels_placeholder, n_mixtures)
