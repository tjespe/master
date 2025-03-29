# USING THE RUGARCH PACKAGE TO FIT A REALIZED GARCH MODEL TO THE DATA
# (Not the same approach as in Hansen et al. 2012, but a more traditional GARCH model with realized volatility as an external regressor)
# Load libraries #
library(rugarch)
library(readr)
library(data.table)
library(future.apply) # For parallel processing

# Plan how to parallelize: multiprocess works on Windows/Linux/Mac
plan(multisession, workers = parallel::detectCores() - 1)

# Load data #
capire_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# define what distribution assumption we would like to use
dist_assumption <- "norm"  # set "norm" for normal and "std" for student-t


# Clean data #

# Return data
# Define what columns to keep
return_data <- return_data[,c("Date", "Symbol", "Total.Return", "LogReturn")]
# remove .O at the end of all symbols
return_data$Symbol = gsub("\\.O", "", return_data$Symbol)
# ensure that the Date column is in the correct format
return_data$Date = as.Date(return_data$Date, format = "%Y-%m-%d")

# Capire data/RV data
# Define what coloumns to keep
capire_data <- capire_data[,c("Date", "Symbol", "RV_5")]
# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")

# transform the RV to become log_daily_rv
capire_data$RV_5 = (capire_data$RV_5/100)/252 # annual percentage^2 --> daily decimal^2
capire_data$RV_5 <- log(capire_data$RV_5 + 1e-10) # log transformation
# sort data by Date and Symbol
return_data <- return_data[order(return_data$Symbol, return_data$Date),]
capire_data <- capire_data[order(capire_data$Symbol, capire_data$Date),]

# Merge data on Date and Symbol
data <- merge(return_data, capire_data, by = c("Date", "Symbol"))
data <- data[order(data$Symbol, data$Date),]

# remove rows with NA values
data <- data[complete.cases(data),]


# define training and validation data
training_data <- data[data$Date < as.Date("2022-12-31"),]
test_data <- data[data$Date >= as.Date("2022-12-31"),]


############################################
# Loop trough symbols and fit model
symbols <- unique(data$Symbol)


for (symbol in symbols) {
  symbol_data <- data[data$Symbol == symbol,]
  rv_values <- symbol_data$RV_5
  
  # Percent of zeros
  percent_zeros <- sum(rv_values == 0) / length(rv_values)
  
  cat(paste0("Symbol: ", symbol, " | Percent Zeros in RV: ", round(percent_zeros * 100, 2), "%\n"))
}

# --- Function to Fit Realized GARCH Model for One Symbol --- #
fit_symbol_garch <- function(symbol) {
  print(paste0("Fitting model for symbol: ", symbol))
  
  symbol_data <- data[data$Symbol == symbol,]
  symbol_training_data <- training_data[training_data$Symbol == symbol,]
  symbol_test_data <- test_data[test_data$Symbol == symbol,]

  window_size <- length(symbol_training_data$Date)
  symbol_data <- rbind(symbol_training_data, symbol_test_data)

  returns <- as.numeric(symbol_data$LogReturn)
  rv <- as.numeric(symbol_data$RV_5)
  epsilon <- 1e-8
  rv <- pmax(rv, epsilon)
  dates <- as.Date(symbol_data$Date)

  total_observations <- length(returns)
  forecast_volatility <- rep(NA, total_observations - window_size)
  forecast_dates <- dates[(window_size + 1):total_observations]

  for (i in seq(window_size, total_observations - 1)) {
    returns_train <- returns[1:i]
    RV_train <- rv[1:i]

    logRV_train_lagged <- log(RV_train[-length(RV_train)])
    returns_train_trimmed <- returns_train[-1]

    spec <- ugarchspec(
      variance.model = list(
        model = "sGARCH",
        garchOrder = c(1, 1),
        external.regressors = matrix(logRV_train_lagged, ncol = 1)
      ),
      mean.model = list(
        armaOrder = c(0, 0),
        include.mean = TRUE
      ),
      distribution.model = dist_assumption
    )

    fit <- ugarchfit(spec = spec, data = returns_train_trimmed, solver = "hybrid")

    logRV_forecast <- log(RV_train[length(RV_train)])

    forecast <- ugarchforecast(fit, n.ahead = 1, external.forecasts = list(vexdata = logRV_forecast))

    forecast_volatility[i - window_size + 1] <- sigma(forecast)

    if ((i - window_size + 1) %% 10 == 0) {
      print(paste0("Symbol: ", symbol, " | Progress: ", round(100 * (i - window_size + 1) / (total_observations - window_size), 2), "%"))
    }
  }

  forecast_results <- data.frame(
    Date = forecast_dates,
    Symbol = symbol,
    Forecast_Volatility = forecast_volatility,
    Mean = 0,
    LogReturn = symbol_data$LogReturn[(window_size + 1):total_observations]
  )

  return(forecast_results)
}

# --- Parallel execution across symbols --- #
results_list <- future_lapply(symbols, fit_symbol_garch)

# Combine results
final_results <- rbindlist(results_list)

# Save results to CSV
write.csv(final_results, paste0("~/Masterv3/master/Code/predictions/realized_garch_forecast_", dist_assumption, ".csv"), row.names = FALSE)

print("Done")


