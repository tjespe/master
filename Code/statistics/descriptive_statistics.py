
#%% Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys

from shared.processing import get_lstm_train_test_new


#%% 
# Load and preprocess data
processed_data = get_lstm_train_test_new()

# Extract train, validation, and test datasets
train_data = processed_data.train
validation_data = processed_data.validation
test_data = processed_data.test

# %%
processed_data
# %%
