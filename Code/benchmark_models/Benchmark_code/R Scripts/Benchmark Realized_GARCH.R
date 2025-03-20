# Following the Hansen et al. (2012), (2014) papers, we will use the realized GARCH model to forecast volatility.

# Load libraries #
library(realized) # TURNS OUT THIS PACKAGE IS NOT AVAILABLE ON CRAN, SO IT IS NOT POSSIBLE TO INSTALL IT
library(readr)

# Load data #
capire_data = read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data = read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# Clean data #

# Return data
# Remove all coloumns except Date, Symbol, Total Return and LogReturn
return_data = return_data[,c("Date", "Symbol", "TotalReturn", "LogReturn")]
# remove .O at the end of all symbols
return_data$Symbol = gsub("\\.O", "", return_data$Symbol)
# ensure that the Date column is in the correct format
return_data$Date = as.Date(return_data$Date, format = "%Y-%m-%d")

# Capire data/RV data
# Remove all coloumns except Date, Symbol and RV_5
capire_data = capire_data[,c("Date", "Symbol", "RV_5")]
# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")
# transform the RV to become daily_rv
capire_data$RV_5 = (capire_data$RV_5/100)/252

# sort data by Date and Symbol
return_data = return_data[order(return_data$Date, return_data$Symbol),]
capire_data = capire_data[order(capire_data$Date, capire_data$Symbol),]

# Merge data on Date and Symbol
data = merge(return_data, capire_data, by = c("Date", "Symbol"))

# define training and validation data
training_data = data[data$Date < as.Date("2021-12-31"),]
validation_data = data[data$Date >= as.Date("2021-12-31") & data$Date <= as.Date("2023-12-31"),]

# Loop trough symbols and fit model
symbols = unique(data$Symbol)

# inititialize list to store results
results = list()

for (symbol in symbols) {
    print(paste0("Fitting model for symbol: ", symbol))
    # filter data for symbol
    symbol_data = data[data$Symbol == symbol,]
    symbol_training_data = training_data[training_data$Symbol == symbol,]
    symbol_validation_data = validation_data[validation_data$Symbol == symbol,]
    
    # define initial window size
    window_size = length(symbol_training_data$Date)

    # concatenate training and validation data
    symbol_data = rbind(symbol_training_data, symbol_validation_data)

    returns = as.numeric(symbol_data$LogReturn)
    rv = as.numeric(symbol_data$RV_5)
    dates = as.Date(symbol_data$Date)

    total_observations = length(returns)
    forecast_volatility <- rep(NA, total_obs - initial_window_size)
    forecast_dates <- dates[(initial_window_size + 1):total_obs] 

    for (i in seq(initial_window_size, total_obs - 1)) {
        # define expanding window
        returns_train = returns[1:i]
        rv_train = rv[1:i]

        # fit realized garch model
        rgarch_model <- realizedGARCH(r = returns_train, realized = RV_train)
        rgarch_forecast <- predict(rgarch_model, n.ahead = 1)

        # store forecasted volatility
        forecast_volatility[i - initial_window_size + 1] <- rgarch_forecast$vol

        # Progress indicator
        if ((i - initial_window_size + 1) %% 10 == 0) {
            print(paste0("Progress: ", round(100 * (i - initial_window_size + 1) / (total_obs - initial_window_size), 2), "%"))
        }
    }


    # store results
    # === Combine results into a forecast data frame ===
    forecast_results <- data.frame(
    Date = forecast_dates,
    Forecast_Volatility = forecast_volatility,
    Forecast_Realized = forecast_realized,
    Mean = 0
)

    results[[symbol]] = forecast_results
}

# write results to csv
write.csv(results, "~/Masterv3/master/Code/predictions/realized_garch_forecast.csv")


