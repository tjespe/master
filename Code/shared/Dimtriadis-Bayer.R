# install.packages("esback")  # Only once
library(esback)
library(dplyr)


# get return data for the test set
return_data <- read.csv("~/Masterv4/master/Code/data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv")
# filter only for test set, from 2023-01-03 to 2024-03-28
return_data <- return_data[return_data$Date >= "2023-01-03" & return_data$Date <= "2024-03-28", ] 
# return_data <- return_data[return_data$Date >= "2005-02-15" & return_data$Date <= "2024-03-28", ]
# remove .O at the end of the Symbol for the return data
return_data$Symbol <- gsub("\\.O$", "", return_data$Symbol)

# load predictions from different models

# comment out the ones we dont have predictions for yet

############# Transformer #############
transformer_RV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/.csv")
transformer_IV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/.csv")
transformer_RV_IV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/transformer_mdn_predictions_stocks_vrv-and-ivol_ensemble.csv")

############ LSTM ###################
lstm_RV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/.csv")
lstm_IV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/lstm_mdn_predictions_stocks_vivol-final_ensemble.csv")
lstm_RV_IV_ensemble <- read.csv("~/Masterv4/master/Code/predictions/lstm_mdn_predictions_stocks_vrv-and-ivol-final_ensemble.csv")
lstm_enire_set <- read.csv("~/Masterv4/master/Code/predictions/temporary_test_lstm_mdn.csv")
########## GARCH MODELS ###########
garch_norm <- read.csv("~/Masterv4/master/Code/predictions/.csv")
garch_t <- read.csv("~/Masterv4/master/Code/predictions/.csv")
rv_garch <- read.csv("~/Masterv4/master/Code/predictions/.csv")
ar_garch_norm <- read.csv("~/Masterv4/master/Code/predictions/.csv")
ar_garch_t <- read.csv("~/Masterv4/master/Code/predictions/.csv")
egarch <- read.csv("~/Masterv4/master/Code/predictions/.csv")

######### HAR ##############
har <- read.csv("~/Masterv4/master/Code/predictions/.csv")
harq <- read.csv("~/Masterv4/master/Code/predictions/.csv")

########### BOOSTERS ###########
catboost_RV <- read.csv("~/Masterv4/master/Code/predictions/CatBoost_RV.csv")
catboost_IV <- read.csv("~/Masterv4/master/Code/predictions/CatBoost_IV.csv")
catboost_RV_IV <- read.csv("~/Masterv4/master/Code/predictions/CatBoost_RV_IV.csv")

xgboost_RV <- read.csv("~/Masterv4/master/Code/predictions/XGBoost_RV.csv")
xgboost_IV <- read.csv("~/Masterv4/master/Code/predictions/XGBoost_IV.csv")
xgboost_RV_IV <- read.csv("~/Masterv4/master/Code/predictions/XGBoost_RV_IV.csv")

lightgbm_RV <- read.csv("~/Masterv4/master/Code/predictions/LightGBM_RV.csv")
lightgbm_IV <- read.csv("~/Masterv4/master/Code/predictions/LightGBM_IV.csv")
lightgbm_RV_IV <- read.csv("~/Masterv4/master/Code/predictions/LightGBM_RV_IV.csv")

########## DB ###########
DB_RV <- read.csv("~/Masterv4/master/Code/predictions/.csv")
DB_IV <- read.csv("~/Masterv4/master/Code/predictions/.csv")
DB_RV_IV <- read.csv("~/Masterv4/master/Code/predictions/.csv")

#######################################################################################################
 # TEST MODELS
#######################################################################################################


##### Transformers and LSTM ####


# Model list
model_list <- list(
 # "LSTM_RV" = lstm_RV_ensemble,
 "LSTM_IV" = lstm_IV_ensemble,
 "LSTM_RV_IV" = lstm_RV_IV_ensemble,
 # "LSTM_temp" = lstm_enire_set
 # "Transformer_RV" = transformer_RV_ensemble,
  #"Transformer_IV" = transformer_IV_ensemble,
 "Transformer_RV_IV" = transformer_RV_IV_ensemble
)

# Alpha levels
alphas <- c(0.01, 0.025, 0.05)


# Initialize result table
results_lstm_transformers <- data.frame()

for (model_name in names(model_list)) {
  model_data <- model_list[[model_name]]
  symbols <- unique(model_data$Symbol)

  for (alpha in alphas) {
    pass_count <- 0
    fail_count <- 0
    
    # cat("Quantile:", alpha)

    for (sym in symbols) {
      # Get symbol-specific data
      model_sym_data <- model_data[model_data$Symbol == sym, ]
      return_sym_data <- return_data[return_data$Symbol == sym, ]
      

      # Ensure data aligns
      if (nrow(model_sym_data) != nrow(return_sym_data)) next
      if (anyNA(model_sym_data) || anyNA(return_sym_data)) next
      


      # Extract variables
      r <- return_sym_data$LogReturn
      q <- model_sym_data[[ifelse(alpha == 0.01, "LB_98", ifelse(alpha == 0.025, "LB_95", "LB_90"))]]
      e <- model_sym_data[[ifelse(alpha == 0.01, "ES_99", ifelse(alpha == 0.025, "ES_97.5", "ES_95"))]]
      
     # cat("Symbol:", sym, "| lengths ??? r:", length(r), " q:", length(q), " e:", length(e), "\n")
     # cat("Classes ??? r:", class(r), " q:", class(q), " e:", class(e), "\n")
     # cat("Any NA ??? r:", anyNA(r), " q:", anyNA(q), " e:", anyNA(e), "\n")

      # Run ESR test
      result <- tryCatch({
        esr_backtest(
          r = r,
          q = q,
          e = e,
          alpha = alpha,
          version = 1,  # Strict ESR
          cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
        )
      }, error = function(e) {
       # cat("??? Error for Symbol:", sym, "| Alpha:", alpha, "\n")
       # cat("   ??? Error message:", conditionMessage(e), "\n\n")
        return(NULL)})

      # Extract p-value and count pass/fail
      if (!is.null(result)) {
        pval <- result$pvalue_twosided_asymptotic
        if (!is.null(pval)) {
          cat("Symbol:", sym, " | Alpha:", alpha, " | p-value:", pval, "\n")
          if (pval >= 0.05) {
            pass_count <- pass_count + 1
          } else {
            fail_count <- fail_count + 1
          }
        }
      }
    }

    total <- pass_count + fail_count
    fail_rate <- ifelse(total > 0, fail_count / total, NA)

    # Store in results table
    results_lstm_transformers <- rbind(results_lstm_transformers, data.frame(
      Model = model_name,
      Alpha = alpha,
      SymbolsTested = total,
      Passed = pass_count,
      Failed = fail_count,
      FailRate = round(fail_rate, 3)
    ))
  }
}

# Display results
print(results_lstm_transformers)

#### GARCH ####


#### BOOSTERS ####
# Model list
model_list <- list(
  "CatBoost_RV" = catboost_RV,
  "CatBoost_IV" = catboost_IV,
  "CatBoost_RV_IV" = catboost_RV_IV,
  "XGBoost_RV" = xgboost_RV,
  "XGBoost_IV" = xgboost_IV,
  "XGBoost_RV_IV" = xgboost_RV_IV,
  "LightGBM_RV" = lightgbm_RV,
  "LightGBM_IV" = lightgbm_IV,
  "LightGBM_RV_IV" = lightgbm_RV_IV
)

# Alpha levels
alphas <- c(0.01, 0.025, 0.05)


# Initialize result table
results_boosters <- data.frame()

for (model_name in names(model_list)) {
  model_data <- model_list[[model_name]]
  symbols <- unique(model_data$Symbol)

  for (alpha in alphas) {
    pass_count <- 0
    fail_count <- 0
    
    # cat("Quantile:", alpha)

    for (sym in symbols) {
      # Get symbol-specific data
      model_sym_data <- model_data[model_data$Symbol == sym, ]
      return_sym_data <- return_data[return_data$Symbol == sym, ]
      

      # Ensure data aligns
      if (nrow(model_sym_data) != nrow(return_sym_data)) next
      if (anyNA(model_sym_data) || anyNA(return_sym_data)) next
      


      # Extract variables
      r <- return_sym_data$LogReturn
      q <- model_sym_data[[ifelse(alpha == 0.01, "Quantile_0.010", ifelse(alpha == 0.025, "Quantile_0.025", "Quantile_0.050"))]]
      e <- model_sym_data[[ifelse(alpha == 0.01, "ES_0.010", ifelse(alpha == 0.025, "ES_0.025", "ES_0.050"))]]
      
     # cat("Symbol:", sym, "| lengths ??? r:", length(r), " q:", length(q), " e:", length(e), "\n")
     # cat("Classes ??? r:", class(r), " q:", class(q), " e:", class(e), "\n")
     # cat("Any NA ??? r:", anyNA(r), " q:", anyNA(q), " e:", anyNA(e), "\n")

      # Run ESR test
      result <- tryCatch({
        esr_backtest(
          r = r,
          q = q,
          e = e,
          alpha = alpha,
          version = 1,  # Strict ESR
          cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
        )
      }, error = function(e) {
       # cat("??? Error for Symbol:", sym, "| Alpha:", alpha, "\n")
       # cat("   ??? Error message:", conditionMessage(e), "\n\n")
        return(NULL)})

      # Extract p-value and count pass/fail
      if (!is.null(result)) {
        pval <- result$pvalue_twosided_asymptotic
        if (!is.null(pval)) {
          cat("Symbol:", sym, " | Alpha:", alpha, " | p-value:", pval, "\n")
          if (pval >= 0.05) {
            pass_count <- pass_count + 1
          } else {
            fail_count <- fail_count + 1
          }
        }
      }
    }

    total <- pass_count + fail_count
    fail_rate <- ifelse(total > 0, fail_count / total, NA)

    # Store in results table
    results_boosters <- rbind(results_boosters, data.frame(
      Model = model_name,
      Alpha = alpha,
      SymbolsTested = total,
      Passed = pass_count,
      Failed = fail_count,
      FailRate = round(fail_rate, 3)
    ))
  }
}

# Display results
print(results_boosters)

### DB ####

# Model list

model_list <- list(
  "DB_RV" = DB_RV,
  "DB_IV" = DB_IV,
  "DB_RV_IV" = DB_RV_IV
)

# Alpha levels
alphas <- c(0.01, 0.025, 0.05)


# Initialize result table
results_DB <- data.frame()

for (model_name in names(model_list)) {
  model_data <- model_list[[model_name]]
  symbols <- unique(model_data$Symbol)

  for (alpha in alphas) {
    pass_count <- 0
    fail_count <- 0
    
    # cat("Quantile:", alpha)

    for (sym in symbols) {
      # Get symbol-specific data
      model_sym_data <- model_data[model_data$Symbol == sym, ]
      return_sym_data <- return_data[return_data$Symbol == sym, ]
      

      # Ensure data aligns
      if (nrow(model_sym_data) != nrow(return_sym_data)) next
      if (anyNA(model_sym_data) || anyNA(return_sym_data)) next
      


      # Extract variables
      r <- return_sym_data$LogReturn
      q <- model_sym_data[[ifelse(alpha == 0.01, "DB_RV_set_0.01", ifelse(alpha == 0.025, "DB_RV_set_0.025", "DB_RV_set_0.05"))]]
      e <- model_sym_data[[ifelse(alpha == 0.01, "DB_RV_set_ES_0.01", ifelse(alpha == 0.025, "DB_RV_set_ES_0.025", "DB_RV_set_ES_0.05"))]]
      
     # cat("Symbol:", sym, "| lengths ??? r:", length(r), " q:", length(q), " e:", length(e), "\n")
     # cat("Classes ??? r:", class(r), " q:", class(q), " e:", class(e), "\n")
     # cat("Any NA ??? r:", anyNA(r), " q:", anyNA(q), " e:", anyNA(e), "\n")

      # Run ESR test
      result <- tryCatch({
        esr_backtest(
          r = r,
          q = q,
          e = e,
          alpha = alpha,
          version = 1,  # Strict ESR
          cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
        )
      }, error = function(e) {
       # cat("??? Error for Symbol:", sym, "| Alpha:", alpha, "\n")
       # cat("   ??? Error message:", conditionMessage(e), "\n\n")
        return(NULL)})

      # Extract p-value and count pass/fail
      if (!is.null(result)) {
        pval <- result$pvalue_twosided_asymptotic
        if (!is.null(pval)) {
          cat("Symbol:", sym, " | Alpha:", alpha, " | p-value:", pval, "\n")
          if (pval >= 0.05) {
            pass_count <- pass_count + 1
          } else {
            fail_count <- fail_count + 1
          }
        }
      }
    }

    total <- pass_count + fail_count
    fail_rate <- ifelse(total > 0, fail_count / total, NA)

    # Store in results table
    results_DB <- rbind(results_DB, data.frame(
      Model = model_name,
      Alpha = alpha,
      SymbolsTested = total,
      Passed = pass_count,
      Failed = fail_count,
      FailRate = round(fail_rate, 3)
    ))
  }
}

# Display results
print(results_DB)

#### HAR ####

# Model list
model_list <- list(
  "HAR" = har,
  "HARQ" = harq
)

# Alpha levels
alphas <- c(0.01, 0.025, 0.05)


# Initialize result table
results_HAR <- data.frame()

for (model_name in names(model_list)) {
  model_data <- model_list[[model_name]]
  symbols <- unique(model_data$Symbol)

  for (alpha in alphas) {
    pass_count <- 0
    fail_count <- 0
    
    # cat("Quantile:", alpha)

    for (sym in symbols) {
      # Get symbol-specific data
      model_sym_data <- model_data[model_data$Symbol == sym, ]
      return_sym_data <- return_data[return_data$Symbol == sym, ]
      

      # Ensure data aligns
      if (nrow(model_sym_data) != nrow(return_sym_data)) next
      if (anyNA(model_sym_data) || anyNA(return_sym_data)) next
      


      # Extract variables
      r <- return_sym_data$LogReturn
      q <- model_sym_data[[ifelse(alpha == 0.01, "LB_98", ifelse(alpha == 0.025, "LB_95", "LB_90"))]]
      e <- model_sym_data[[ifelse(alpha == 0.01, "ES_99", ifelse(alpha == 0.025, "ES_97.5", "ES_95"))]]
      
     # cat("Symbol:", sym, "| lengths ??? r:", length(r), " q:", length(q), " e:", length(e), "\n")
     # cat("Classes ??? r:", class(r), " q:", class(q), " e:", class(e), "\n")
     # cat("Any NA ??? r:", anyNA(r), " q:", anyNA(q), " e:", anyNA(e), "\n")

      # Run ESR test
      result <- tryCatch({
        esr_backtest(
          r = r,
          q = q,
          e = e,
          alpha = alpha,
          version = 1,  # Strict ESR
          cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
        )
      }, error = function(e) {
       # cat("??? Error for Symbol:", sym, "| Alpha:", alpha, "\n")
       # cat("   ??? Error message:", conditionMessage(e), "\n\n")
        return(NULL)})

      # Extract p-value and count pass/fail
      if (!is.null(result)) {
        pval <- result$pvalue_twosided_asymptotic
        if (!is.null(pval)) {
          cat("Symbol:", sym, " | Alpha:", alpha, " | p-value:", pval, "\n")
          if (pval >= 0.05) {
            pass_count <- pass_count + 1
          } else {
            fail_count <- fail_count + 1
          }
        }
      }
    }

    total <- pass_count + fail_count
    fail_rate <- ifelse(total > 0, fail_count / total, NA)

    # Store in results table
    results_HAR <- rbind(results_HAR, data.frame(
      Model = model_name,
      Alpha = alpha,
      SymbolsTested = total,
      Passed = pass_count,
      Failed = fail_count,
      FailRate = round(fail_rate, 3)
    ))
  }
}

# Display results
print(results_HAR)


###########################
# Combine all results into one table and display
###########################
# all_results <- rbind(results_lstm_transformers, results_boosters, results_DB, results_HAR)
all_results <- rbind(results_lstm_transformers, results_boosters) 
all_results <- all_results %>% arrange(Model, Alpha)
all_results$Model <- factor(all_results$Model, levels = unique(all_results$Model))
all_results$Alpha <- factor(all_results$Alpha, levels = c(0.01, 0.025, 0.05), labels = c("1%", "2.5%", "5%"))
all_results$FailRate <- round(all_results$FailRate * 100, 2)  # Convert to percentage


# Display the final results table
print(all_results)

# display it as a nice table
library(knitr)
library(kableExtra)

all_results %>%
  kable("html", escape = F, col.names = c("Model", "Alpha", "Symbols Tested", "Passed", "Failed", "Fail Rate (%)")) %>%
  kable_styling(full_width = F, position = "left") %>%
  column_spec(1, bold = T) %>%
  column_spec(2, bold = T) %>%
  column_spec(3, bold = T) %>%
  column_spec(4, bold = T) %>%
  column_spec(5, bold = T) %>%
  column_spec(6, bold = T)


