# Define parameters
LOOKBACK_DAYS = 30
SUFFIX = ""  # Use "_stocks" for the single stocks or "" for S&P500 only
TEST_ASSET = "S&P"
DATA_PATH = "data/spx_garch.csv"
TRAIN_VALIDATION_SPLIT = "2021-12-31"
VALIDATION_TEST_SPLIT = "2023-12-31"

# Other
# Not necessary anymore after we stopped using Git LFS
# import _fix_path
