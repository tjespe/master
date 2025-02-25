# %%
# Define parameters
from settings import LOOKBACK_DAYS, SUFFIX, DATA_PATH, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT, TEST_ASSET
MODEL_NAME = f"VAE_v1_{LOOKBACK_DAYS}_days{SUFFIX}"
# %%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from shared.processing import get_lstm_train_test_new


