import os
from typing import Literal

BASEDIR = os.path.abspath(os.path.dirname(__file__))

# Define parameters
LOOKBACK_DAYS = 30
SUFFIX = "_stocks"  # Use "_stocks" for the single stocks or "" for S&P500 only
DATA_PATH = f"{BASEDIR}/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv"
TRAIN_VALIDATION_SPLIT = "2021-12-31"
VALIDATION_TEST_SPLIT = "2022-12-31"

# Only used in old single stock model comparison
TEST_ASSET = "AAPL"

# Define which test set to use (either "test" or "validation")
TEST_SET: Literal["test", "validation"] = "test"

# Other
# Not necessary anymore after we stopped using Git LFS
# import _fix_path
