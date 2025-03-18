# HAR / HARQ model using Realized Volatility as the dependent variable following Bollerslev et al 2016 approach

# load required libraries
library(dplyr)
library(zoo)
library(data.table)

############################################
# Load data #
capire_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# Define wether to use HAR or HARQ, include_RQ = TRUE for HARQ
include_RQ <- FALSE

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
capire_data <- capire_data[,c("Date", "Symbol", "RV_5", "RQ_5")]
# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")

# transform the RV and RQ to daily
capire_data$RV_5 = (capire_data$RV_5/100)/252  # annual percentage^2 --> daily decimal^2
capire_data$RQ_5 = (capire_data$RQ_5/100^4)/252^2 # annual percentage^4 --> daily decimal^4

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

# Loop trough symbols and create features
symbols <- unique(data$Symbol)
results <- list()

for (symbol in symbols) {
    cat("Processing Symbol:", symbol, "\n")

    symbol_data <- data[data$Symbol == symbol,]
    symbol_training_data <- training_data[training_data$Symbol == symbol,]
    symbol_validation_data <- validation_data[validation_data$Symbol == symbol,]

    # define initial window size
    window_size <- length(symbol_training_data$Date)

    # concatenate training and validation data
    symbol_data <- rbind(symbol_training_data, symbol_validation_data)

    # create features
    symbol_data$RV_lag1 <- dplyr::lag(symbol_data$RV_5, 1)
    symbol_data$RV_lag5 <- dplyr::lag(zoo::rollmean(symbol_data$RV_5, 5, align = "right", fill = NA), 1)
    symbol_data$RV_lag22 <- dplyr::lag(zoo::rollmean(symbol_data$RV_5, 22, align = "right", fill = NA), 1)

    # Conditionally add RQ features if HARQ is selected
    if (include_RQ) {
        symbol_data$RQ_lag1 <- dplyr::lag(symbol_data$RQ_5, 1)
        symbol_data$RV_lag1_RQ_lag1 <- dplyr::lag(symbol_data$RV_lag1*sqrt(symbol_data$RQ_lag1), 1)
    }

    total_obs <- nrow(symbol_data)

    forecast_values <- rep(NA, total_obs - window_size)
    forecast_dates <- symbol_data$Date[(window_size + 1):total_obs]

    for (i in seq(window_size, total_obs-1)) {
        # Define expanding training window
        train_data <- symbol_data[1:i, ]
        
        # Conditionally add RQ features if HARQ is selected
        if (include_RQ) {
            har_model <- lm(RV_5 ~ RV_lag1 + RV_lag1_RQ_lag1 + RV_lag5 + RV_lag22, data = train_data)
        } else {
            har_model <- lm(RV_5 ~ RV_lag1 + RV_lag5 + RV_lag22, data = train_data)
        }
        # Forecast for i + 1
        forecast_data <- symbol_data[(i + 1), ]
        
        # Generate prediction
        predicted_var <- predict(har_model, newdata = forecast_data)
        predicted_vol <- sqrt(predicted_var)

        forecast_values[i - window_size + 1] <- predicted_vol
        
        # Progress indicator
        if ((i - window_size + 1) %% 20 == 0) {
        cat("Progress:", round(100 * (i - window_size + 1) / (total_obs - window_size), 2), "%\n")
        }
    }

     # Create result dataframe for this symbol
    forecast_df <- data.frame(
        Date = forecast_dates,
        Symbol = symbol,
        HAR_vol_R = forecast_values
    )
    
    # Append to the list
    results[[symbol]] <- forecast_df

}

final_results <- rbindlist(results)

# write results to csv
if (include_RQ) {
    write.csv(final_results, paste0("~/Masterv3/master/Code/predictions/HARQ_R.csv"))
} else {
write.csv(final_results, paste0("~/Masterv3/master/Code/predictions/HAR_R.csv"))
}
