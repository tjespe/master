install.packages("esback")  # Only once
library(esback)

# load predictions from different models as a data frame

############# Transformer #############


############ LSTM ###################

lstm_IV_ensemble <- mydata <- read.csv("../predictions/lstm_mdn_predictions_stocks_vivol-final_ensemble.csv")
########## GARCH ###########


########### BOOSTERS ###########


########## DB ###########




# implement the test
cov_settings <- list(
    sparsity = "nid",
    sigma_est = "scl_sp",
  misspec = TRUE           # Allow for model misspecification (robust covariance)
)

# version specification: 1 = Strict ESR, 2 = Auxilliary ESR, 3 = Strict Interceots
result <- esr_backtest(
  r = r,
  q = q,
  e = e,
  alpha = alpha,
  version = 1,
  cov_config = cov_settings
)