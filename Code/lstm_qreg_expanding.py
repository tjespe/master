# %%
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    Embedding,
    Flatten,
    Concatenate,
    RepeatVector,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Loss

from shared.processing import get_lstm_train_test_new
from shared.conf_levels import format_cl
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    VALIDATION_TEST_SPLIT,
)

# %% PARAMETERS

VERSION = "iv"
MODEL_NAME = f"lstm_qr{SUFFIX}_{VERSION}"
# feature flags (as before)
INCLUDE_TICKERS = False
INCLUDE_10_DAY_IVOL = "iv" in VERSION
INCLUDE_30_DAY_IVOL = "iv" in VERSION
INCLUDE_1MIN_RV = "rv" in VERSION
INCLUDE_5MIN_RV = "rv" in VERSION

# model hyperparams
HIDDEN_UNITS = 60
DROPOUT = 0.0
L2_REG = 1e-4
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None  # if using tickers

# training
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 15
ROLLING_INTERVAL = 30  # days between retrains

# %% DYNAMIC QUANTILE SETUP

# 1) your target confidence levels
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]
# 2) ES levels (upper tail prob)
ES_LEVELS = [1 - (1 - cl) / 2 for cl in CONFIDENCE_LEVELS]
# 3) how many extra quantiles for ES approximation
n_es_points = 5

# build the minimal quantile set (lower & upper bounds)
base_quantiles = sorted(
    [(1 - cl) / 2 for cl in CONFIDENCE_LEVELS]
    + [(1 + cl) / 2 for cl in CONFIDENCE_LEVELS]
)

# add extra small quantiles for ES integration
es_extra = set()
for alpha in ES_LEVELS:
    lower_q = 1 - alpha
    qs = np.linspace(0, lower_q, n_es_points + 1)[1:]
    es_extra.update(qs)

# final sorted, unique list
QUANTILES = sorted(set(base_quantiles).union(es_extra))
N_QUANTILES = len(QUANTILES)

# map each confidence level to its lower-tail quantile
CL_TO_LOWER_Q = {cl: (1 - cl) / 2 for cl in CONFIDENCE_LEVELS}


# %% PINBALL LOSS
class PinballLoss(Loss):
    """Vectorized pinball (quantile) loss for multiple quantiles."""

    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.q = tf.constant(quantiles, dtype=tf.float32)  # shape (N_QUANTILES,)

    def call(self, y_true, y_pred):
        # y_true: (batch, ), y_pred: (batch, N_QUANTILES)
        y_true = tf.expand_dims(y_true, axis=-1)  # (batch,1)
        e = y_true - y_pred  # (batch, N_QUANTILES)
        loss = tf.maximum(self.q * e, (self.q - 1) * e)
        return tf.reduce_mean(loss)


# %% MODEL CONSTRUCTION
def build_lstm_qr(num_features: int, ticker_ids_dim: Optional[int]):
    seq_input = Input(shape=(LOOKBACK_DAYS, num_features), name="seq_input")
    x = seq_input

    if ticker_ids_dim is not None:
        tid_in = Input(shape=(), dtype="int32", name="ticker_input")
        emb = Embedding(
            input_dim=ticker_ids_dim, output_dim=EMBEDDING_DIMENSIONS, name="ticker_emb"
        )(tid_in)
        emb = Flatten()(emb)
        emb = RepeatVector(LOOKBACK_DAYS)(emb)
        x = Concatenate(name="concat_seq_ticker")([seq_input, emb])
        inputs = [seq_input, tid_in]
    else:
        inputs = [seq_input]

    x = LSTM(
        HIDDEN_UNITS, activation="tanh", kernel_regularizer=l2(L2_REG), name="lstm"
    )(x)
    if DROPOUT > 0:
        x = Dropout(DROPOUT, name="dropout")(x)

    for i in range(NUM_HIDDEN_LAYERS):
        x = Dense(
            HIDDEN_UNITS,
            activation="relu",
            kernel_regularizer=l2(L2_REG),
            name=f"dense_{i}",
        )(x)
        if DROPOUT > 0:
            x = Dropout(DROPOUT, name=f"dropout_{i}")(x)

    # final head: one output per quantile
    q_out = Dense(N_QUANTILES, name="quantile_outputs")(x)
    return Model(inputs, q_out, name="LSTM-QR")


# %% ROLLING TRAIN & PREDICT

if __name__ == "__main__":
    print(f"Rolling LSTM-QR: {VERSION}")

    # %%
    # load data
    data = get_lstm_train_test_new(
        include_returns=False,
        include_spx_data=False,
        include_others=False,
        include_beta=False,
        include_fng=False,
        include_garch=False,
        include_industry=False,
        include_fred_md=False,
        include_1min_rv=INCLUDE_1MIN_RV,
        include_5min_rv=INCLUDE_5MIN_RV,
        include_ivol_cols=(["10 Day Call IVOL"] if INCLUDE_10_DAY_IVOL else [])
        + (["Historical Call IVOL"] if INCLUDE_30_DAY_IVOL else []),
    )

    # %%
    # set up rolling retrain
    first_test = pd.to_datetime(VALIDATION_TEST_SPLIT)
    if not ROLLING_INTERVAL:
        ROLLING_INTERVAL = np.inf
    end_date = lambda: first_test + pd.DateOffset(days=ROLLING_INTERVAL)

    all_preds = []
    symbols = []
    dates = []
    true_y = []

    while True:
        test = data.get_test_set_for_date(first_test, end_date())
        if test.y.shape[0] == 0:
            break
        train = data.get_training_set_for_date(first_test)

        print("Train:", train.dates.min(), "to", train.dates.max())
        print("Test:", test.dates.min(), "to", test.dates.max())

        # build & compile
        model = build_lstm_qr(
            train.X.shape[2], data.ticker_ids_dim if INCLUDE_TICKERS else None
        )
        model.compile(
            optimizer=AdamW(learning_rate=LEARNING_RATE, decay=WEIGHT_DECAY),
            loss=PinballLoss(QUANTILES),
        )

        # load or fit
        fname = f"models/rolling/{MODEL_NAME}_{first_test.date()}.weights.h5"
        if os.path.exists(fname):
            model.load_weights(fname)
            print("Loaded existing weights.")
        else:
            model.fit(
                train.X if not INCLUDE_TICKERS else [train.X, train.ticker_ids],
                train.y,
                validation_data=(
                    test.X if not INCLUDE_TICKERS else [test.X, test.ticker_ids],
                    test.y,
                ),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
            )
            model.save_weights(fname)

        # predict
        preds = model.predict(
            test.X if not INCLUDE_TICKERS else [test.X, test.ticker_ids],
            batch_size=BATCH_SIZE,
        )

        # ensure non-crossing quantiles, consistent with Moen et al.'s approach
        preds = np.maximum.accumulate(preds, axis=1)

        all_preds.append(preds)
        symbols += list(test.tickers)
        dates += list(test.dates)
        true_y += list(test.y)

        first_test = end_date() + pd.DateOffset(days=1)
        gc.collect()

    # %%
    # concatenate
    y_pred = np.vstack(all_preds)  # (N_total, N_QUANTILES)
    true_y = np.array(true_y)
    df = pd.DataFrame({"Date": dates, "Symbol": symbols, "True": true_y})

    # %%
    # assign VaR bounds and ES
    quant_arr = np.array(QUANTILES)
    for cl in CONFIDENCE_LEVELS:
        lower_q = CL_TO_LOWER_Q[cl]
        upper_q = 1 - lower_q
        # indices
        i_low = np.where(np.isclose(quant_arr, lower_q))[0][0]
        i_up = np.where(np.isclose(quant_arr, upper_q))[0][0]
        df[f"LB_{format_cl(cl)}"] = y_pred[:, i_low]
        df[f"UB_{format_cl(cl)}"] = y_pred[:, i_up]
        # ES approx = mean of all preds with q <= lower_q
        idxs = np.where(quant_arr <= lower_q)[0]
        df[f"ES_{format_cl(1-lower_q)}"] = y_pred[:, idxs].mean(axis=1)

    # %%
    # Example plot of ES
    df.set_index(["Date", "Symbol"]).xs("AAPL", level="Symbol").sort_index()[
        ["True", "LB_90", "ES_95", "LB_98", "ES_99"]
    ].rename(
        columns={
            "LB_90": "95% VaR",
            "ES_95": "95% ES",
            "LB_98": "99% VaR",
            "ES_99": "99% ES",
        }
    ).plot(
        title="99% VaR and ES for AAPL",
        # Color appropriately
        color=["#ccc", "#ffaaaa", "#ff0000", "#aaaaff", "#0000ff"],
        figsize=(12, 6),
    )

    # %%
    # save
    os.makedirs("predictions", exist_ok=True)
    out_csv = f"predictions/{MODEL_NAME}.csv"
    df.to_csv(out_csv, index=False)
    print("Saved predictions to", out_csv)
