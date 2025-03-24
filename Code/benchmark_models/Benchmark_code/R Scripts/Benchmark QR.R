# Quantile regression model 

library(quantreg)
library(dplyr)
library(zoo)
library(ggplot2)
library(data.table)
library(future)
library(future.apply)

# set up parallel processing
# plan(multisession, workers = parallel::detectCores() - 1)

# Parameters
tau_levels <- c(0.9, 0.95, 0.975, 0.99)            # Quantiles to predict
window_size <- 1500                          # Rolling window size
dependent_var <- "Return"                    # Dependent variable
independent_vars <- c("Volatility", "RV", "RQ")  # Independent variables

# load and clean data
capire_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv")
return_data <- read.csv("~/Masterv3/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")

# remove .O at the end of all symbols
return_data$Symbol = gsub("\\.O", "", return_data$Symbol)
# ensure that the Date column is in the correct format
return_data$Date = as.Date(return_data$Date, format = "%Y-%m-%d")

# ensure that the Date column is in the correct format
capire_data$Date = as.Date(capire_data$Date, format = "%Y-%m-%d")
# transform RV and RQ to become daily
capire_data$RV = (capire_data$RV/100)/252
capire_data$RQ = (capire_data$RQ/100)/252






