# Following the Hansen et al. (2012), (2014) papers, we will use the realized GARCH model to forecast volatility.
#%%
install.packages("rugarch")
install.packages("data.table")
install.packages("readr")
install.packages("future.apply")
install.packages("xts")
# Load libraries #
library(rugarch)
library(readr)
library(data.table)
library(future.apply)
library(xts) # For time series data handling
# %%
# Plan how to parallelize: multiprocess works on Windows/Linux/Mac
plan(multisession, workers = parallel::detectCores() - 1)
# %%
dist_assumption <- "norm" # set "norm" for normal and "std" for student-t
# %%
# Load data #
capire_data = read.csv("~/Master/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data = read.csv("~/Master/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# Clean data #

# Return data
# Remove all coloumns except Date, Symbol, Total Return and LogReturn
return_data = return_data[,c("Date", "Symbol", "LogReturn")]
# remove .O at the end of all symbols
return_data$Symbol = gsub("\\.O", "", return_data$Symbol)
# ensure that the Date column is in the correct format
return_data$Date = as.Date(return_data$Date, format = "%Y-%m-%d")
# remove the symbo DOW from the data
return_data = return_data[return_data$Symbol != "DOW",]

# Capire data/RV data
# Remove all coloumns except Date, Symbol and RV_5
capire_data = capire_data[,c("Date", "Symbol", "RV_5")]
# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")
# transform the RV to become decimal
capire_data$RV_5 = (capire_data$RV_5/10000)
# no need to log RV data as th realGARCH model will do that internally
# remove the symbo DOW from the data
capire_data = capire_data[capire_data$Symbol != "DOW",]

# sort data by Date and Symbol
return_data = return_data[order(return_data$Date, return_data$Symbol),]
capire_data = capire_data[order(capire_data$Date, capire_data$Symbol),]

# Merge data on Date and Symbol
data = merge(return_data, capire_data, by = c("Date", "Symbol"))

# define training and test data
training_data = data[data$Date >= as.Date("2003-01-02") & data$Date < as.Date("2019-12-31"),]
test_data = data[data$Date >= as.Date("2019-12-31"),]

# Loop trough symbols and fit model
symbols = unique(data$Symbol)

#%%

# --- Function to Fit Realized GARCH Model for One Symbol --- #
fit_symbol_garch <- function(symbol) {
  print(paste0("Fitting model for symbol: ", symbol))

  symbol_data <- data[data$Symbol == symbol, ]
  symbol_training_data <- training_data[training_data$Symbol == symbol, ]
  symbol_test_data <- test_data[test_data$Symbol == symbol, ]

  window_size <- length(symbol_training_data$Date)
  symbol_data <- rbind(symbol_training_data, symbol_test_data)

  returns <- as.numeric(symbol_data$LogReturn)
  rv <- as.numeric(symbol_data$RV_5)
  epsilon <- 1e-8
  rv <- pmax(rv, epsilon)
  dates <- as.Date(symbol_data$Date)

  # make two xts series in order to use the ugarchfit function
  returns_xts <- xts(returns, order.by = dates)
  rv_xts      <- xts(rv, order.by = dates)

  total_observations <- length(returns)
  forecast_volatility <- rep(NA, total_observations - window_size)
  forecast_dates <- dates[(window_size + 1):total_observations]

  for (i in seq(window_size, total_observations - 1)) {
    # 1) expanding-window data. # no need to lag returns and RV as the realGARCH model will do that internally
    returns_train_xts <- returns_xts[1:i]
    rv_train_xts      <- rv_xts[1:i]

    # 2) RealGARCH(1,1) spec
    spec <- ugarchspec(
        variance.model = list(model = "realGARCH",
                            garchOrder = c(1,1)),
        mean.model     = list(armaOrder  = c(0,0),
                            include.mean = TRUE),
        distribution.model = dist_assumption
    )

    # 3) fit with realizedVol = RV_train
    fit <- tryCatch(
        ugarchfit(spec, data = returns_train_xts, realizedVol = rv_train_xts,
                solver = "hybrid"),
        error = function(e) {
        message("Fit error at i=", i, ": ", e$message)
        return(NULL)
        }
    )
    if (is.null(fit)) next

    # 4) next-day forecast: pass rv[i] (the last observed RV) as the next realizedVol
    next_date <- index(returns_xts)[i]  # date of observation i
    fc <- ugarchforecast(
    fit, n.ahead=1,
    realizedVol = xts(rv_xts[i], order.by = next_date)
  )
    forecast_volatility[i - window_size + 1] <- sigma(fc)
  

    # 5) progress
    if (((i - window_size + 1) %% 10) == 0) {
        pct <- round(100 * (i - window_size + 1) / (total_observations - window_size), 1)
        message(symbol, ": ", pct, "% done")
        }
    }

  forecast_results <- data.frame(
    Date = forecast_dates,
    Symbol = symbol,
    Forecast_Volatility_real_garch = forecast_volatility,
    Mean_real_garch = 0,
    LogReturn = symbol_data$LogReturn[(window_size + 1):total_observations]
  )

  return(forecast_results)
}
# %%
# --- Parallel execution across symbols --- #
results_list <- future_lapply(symbols, fit_symbol_garch)
# %%
# Combine results
final_results <- rbindlist(results_list)
# %%

# write results to csv including the distribution assumption at the end of the file name
write.csv(final_results, paste0("~/Master/master/Code/predictions/realized_garch_forecast_actual", dist_assumption, ".csv"))


