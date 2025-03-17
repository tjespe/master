# USING THE RUGARCH PACKAGE TO FIT A REALIZED GARCH MODEL TO THE DATA
# Load libraries #
library(rugarch)
library(readr)
library(data.table)

# Load data #
capire_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# define what distribution assumption we would like to use
dist_assumption <- "norm"  # set "norm" for normal and "std" for student-t


# Clean data #

# Return data
# Remove all coloumns except Date, Symbol, Total Return and LogReturn
return_data <- return_data[,c("Date", "Symbol", "Total.Return", "LogReturn")]
# remove .O at the end of all symbols
return_data$Symbol = gsub("\\.O", "", return_data$Symbol)
# ensure that the Date column is in the correct format
return_data$Date = as.Date(return_data$Date, format = "%Y-%m-%d")

# Capire data/RV data
# Remove all coloumns except Date, Symbol and RV_5
capire_data <- capire_data[,c("Date", "Symbol", "RV_5")]
# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")
# transform the RV to become daily_rv
capire_data$RV_5 = (capire_data$RV_5/100)/252

# sort data by Date and Symbol
return_data <- return_data[order(return_data$Symbol, return_data$Date),]
capire_data <- capire_data[order(capire_data$Symbol, capire_data$Date),]

# Merge data on Date and Symbol
data <- merge(return_data, capire_data, by = c("Date", "Symbol"))
data <- data[order(data$Symbol, data$Date),]

# remove rows with NA values
data <- data[complete.cases(data),]


# define training and validation data
training_data <- data[data$Date < as.Date("2021-12-31"),]
validation_data <- data[data$Date >= as.Date("2021-12-31") & data$Date <= as.Date("2023-12-31"),]


############################################
# Loop trough symbols and fit model
symbols <- unique(data$Symbol)

# remove all sybols except TRV, UNH, V, VZ, WMT
symbols <- symbols[symbols %in% c("TRV", "UNH", "V", "VZ", "WMT")]


for (symbol in symbols) {
  symbol_data <- data[data$Symbol == symbol,]
  rv_values <- symbol_data$RV_5
  
  # Percent of zeros
  percent_zeros <- sum(rv_values == 0) / length(rv_values)
  
  cat(paste0("Symbol: ", symbol, " | Percent Zeros in RV: ", round(percent_zeros * 100, 2), "%\n"))
}


# inititialize list to store results
results <- list()

for (symbol in symbols) {
    print(paste0("Fitting model for symbol: ", symbol))
    # filter data for symbol
    symbol_data <- data[data$Symbol == symbol,]
    symbol_training_data <- training_data[training_data$Symbol == symbol,]
    symbol_validation_data <- validation_data[validation_data$Symbol == symbol,]
    
    # define initial window size
    window_size <- length(symbol_training_data$Date)

    # concatenate training and validation data
    symbol_data <- rbind(symbol_training_data, symbol_validation_data)

    returns <- as.numeric(symbol_data$LogReturn)
    rv <- as.numeric(symbol_data$RV_5)
    #if there are 0 values, replace with very small number 
    epsilon <- 1e-8
    rv <- pmax(rv, epsilon)
    dates <- as.Date(symbol_data$Date)

    total_observations <- length(returns)
    forecast_volatility <- rep(NA, total_observations - window_size)
    forecast_dates <- dates[(window_size + 1):total_observations] 

    for (i in seq(window_size, total_observations - 1)) {
        # define expanding window
        returns_train <- returns[1:i]
        RV_train <- rv[1:i]

        # external regressors
        logRV_train_lagged <- log(RV_train[-length(RV_train)])
        returns_train_trimmed <- returns_train[-1]  # Align

        # Model specification for each iteration
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

        # Fit the model on current expanding window
        fit <- ugarchfit(spec = spec, data = returns_train_trimmed, solver = "hybrid")

        # Prepare external regressor for the next period forecast
        # Using log RV of the current last training observation
        logRV_forecast <- log(RV_train[length(RV_train)])
        
        # Forecast 1-step ahead
        forecast <- ugarchforecast(fit, n.ahead = 1,
                                    external.forecasts = list(vexdata = logRV_forecast))
        
        # Extract the 1-step-ahead conditional variance forecast (sigma^2)
        forecast_volatility[i - window_size + 1] <- sigma(forecast)

        # Progress indicator
        if ((i - window_size + 1) %% 10 == 0) {
            print(paste0("Progress: ", round(100 * (i - window_size + 1) / (total_observations - window_size), 2), "%"))
        }
    }


    # store results
    # === Combine results into a forecast data frame ===
    # Create forecast results data frame
    forecast_results <- data.frame(
    Date = forecast_dates,
    Symbol = symbol,
    Forecast_Volatility = forecast_volatility,
    Mean = 0
    )

    results[[symbol]] = forecast_results
}

final_results <- rbindlist(results)

# write results to csv
write.csv(final_results, paste0("~/Masterv3/master/Code/predictions/realized_garch_forecast_", dist_assumption, ".csv"))



