# install.packages("esback")  # Only once
library(esback)
library(dplyr)

# UPDATE BASEPATH TO YOUR FILE PATH
base_path_return_data <- "~/Masterv4/master/Code/data/dow_jones/processed_data"
base_path_predictions <- "~/Masterv4/master/Code/predictions"

test_set_start_date <- "2019-12-31"


# get return data for the test set
return_data <- read.csv(file.path(base_path_return_data, "dow_jones_stocks_1990_to_today_19022025_cleaned_garch.csv"))
# filter only for test set, from 2023-01-03 to 2024-03-28
return_data <- return_data[return_data$Date >= test_set_start_date & return_data$Date <= "2024-03-28", ] 
# return_data <- return_data[return_data$Date >= "2005-02-15" & return_data$Date <= "2024-03-28", ]
# remove .O at the end of the Symbol for the return data
return_data$Symbol <- gsub("\\.O$", "", return_data$Symbol)

# load predictions from different models

# comment out the ones we dont have predictions for yet

############# Transformer #############

# non-rolling
transformer_RV_ensemble     <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vrv_test_ensemble.csv"))
transformer_IV_ensemble     <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vivol_test_ensemble.csv"))
transformer_RV_IV_ensemble  <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vrv-and-ivol_test_ensemble.csv"))

# rolling
#transformer_RV_ensemble_rolling     <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_stocks_vrv-final-rolling_ensemble_test.csv"))
#transformer_IV_ensemble_rolling     <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_stocks_vivol-final-rolling_ensemble_test.csv"))
#transformer_RV_IV_ensemble_rolling  <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_stocks_vrv-and-ivol-final-rolling_ensemble_test.csv"))

############ LSTM ###################

#non-rolling
lstm_RV_ensemble     <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vrv-final_ensemble.csv"))
lstm_IV_ensemble     <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vivol-final_ensemble.csv"))
lstm_RV_IV_ensemble  <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vrv-and-ivol-final_ensemble.csv"))

#rolling
lstm_RV_ensemble_rolling     <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vrv-final-rolling_test.csv"))
lstm_IV_ensemble_rolling     <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vivol-final-rolling_test.csv"))
lstm_RV_IV_ensemble_rolling  <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vrv-and-ivol-final-rolling_test.csv"))

############## ENSEMBLE COMBINATIONS ###############
#MDN_ensemble_IV_RV <- read.csv(file.path(base_path_predictions, "ensemble_mdn_predictions_stocks_vrv-iv_ensemble.csv"))

########## GARCH MODELS ###########
garch_norm     <- read.csv(file.path(base_path_predictions, "GARCH_preds_enriched.csv"))
#garch_t        <- read.csv(file.path(base_path_predictions, "garch_predictions_student_t.csv"))
#rv_garch       <- read.csv(file.path(base_path_predictions, "realized_garch_forecast_norm.csv"))
#ar_garch_norm  <- read.csv(file.path(base_path_predictions, "predictions_AR(1)-GARCH(1,1)-normal.csv"))
# ar_garch_t   <- read.csv(file.path(base_path, ".csv"))  # Not included because file name is missing
egarch         <- read.csv(file.path(base_path_predictions, "EGARCH_preds_enriched.csv"))


# filter only for test set, from test_set_start_date to 2024-03-28
garch_norm <- garch_norm[garch_norm$Date >= test_set_start_date & garch_norm$Date <= "2024-03-28", ]
#garch_t <- garch_t[garch_t$Date >= test_set_start_date & garch_t$Date <= "2024-03-28", ]
#rv_garch <- rv_garch[rv_garch$Date >= test_set_start_date & rv_garch$Date <= "2024-03-28", ]
#ar_garch_norm <- ar_garch_norm[ar_garch_norm$Date >= test_set_start_date & ar_garch_norm$Date <= "2024-03-28", ]
#ar_garch_t <- ar_garch_t[ar_garch_t$Date >= test_set_start_date & ar_garch_t$Date <= "2024-03-28", ]
egarch <- egarch[egarch$Date >= test_set_start_date & egarch$Date <= "2024-03-28", ]

# remove .O at the end of the Symbol for the garch models
garch_norm$Symbol <- gsub("\\.O$", "", garch_norm$Symbol)
garch_t$Symbol <- gsub("\\.O$", "", garch_t$Symbol)
rv_garch$Symbol <- gsub("\\.O$", "", rv_garch$Symbol)
ar_garch_norm$Symbol <- gsub("\\.O$", "", ar_garch_norm$Symbol)
#ar_garch_t$Symbol <- gsub("\\.O$", "", ar_garch_t$Symbol)
egarch$Symbol <- gsub("\\.O$", "", egarch$Symbol)


######### HAR ##############
har   <- read.csv(file.path(base_path_predictions, "HAR_R.csv"))
harq  <- read.csv(file.path(base_path_predictions, "HARQ_R.csv"))

########### BOOSTERS ###########
# catboost_RV <- read.csv(file.path(base_path, "CatBoost_RV.csv"))
#catboost_IV     <- read.csv(file.path(base_path_predictions, "CatBoost_IV.csv"))
#catboost_RV_IV  <- read.csv(file.path(base_path_predictions, "CatBoost_RV_IV.csv"))

#xgboost_RV      <- read.csv(file.path(base_path_predictions, "XGBoost_RV.csv"))
# xgboost_IV    <- read.csv(file.path(base_path, "XGBoost_IV.csv"))
# xgboost_RV_IV <- read.csv(file.path(base_path, "XGBoost_RV_IV.csv"))

#lightgbm_RV     <- read.csv(file.path(base_path_predictions, "LightGBM_RV.csv"))
#lightgbm_IV     <- read.csv(file.path(base_path_predictions, "LightGBM_IV.csv"))
#lightgbm_RV_IV  <- read.csv(file.path(base_path_predictions, "LightGBM_RV_IV.csv"))

########## DB ###########
# DB_RV     <- read.csv(file.path(base_path, ".csv"))
# DB_IV     <- read.csv(file.path(base_path, ".csv"))
# DB_RV_IV  <- read.csv(file.path(base_path, ".csv"))


# remove ticker "DOW" from all models if it exists
transformer_RV_ensemble <- transformer_RV_ensemble[transformer_RV_ensemble$Symbol != "DOW", ]
transformer_IV_ensemble <- transformer_IV_ensemble[transformer_IV_ensemble$Symbol != "DOW", ]
transformer_RV_IV_ensemble <- transformer_RV_IV_ensemble[transformer_RV_IV_ensemble$Symbol != "DOW", ]
# transformer_RV_ensemble_rolling <- transformer_RV_ensemble_rolling[transformer_RV_ensemble_rolling$Symbol != "DOW", ]
# transformer_IV_ensemble_rolling <- transformer_IV_ensemble_rolling[transformer_IV_ensemble_rolling$Symbol != "DOW", ]
# transformer_RV_IV_ensemble_rolling <- transformer_RV_IV_ensemble_rolling[transformer_RV_IV_ensemble_rolling$Symbol != "DOW", ]
lstm_RV_ensemble <- lstm_RV_ensemble[lstm_RV_ensemble$Symbol != "DOW", ]
lstm_IV_ensemble <- lstm_IV_ensemble[lstm_IV_ensemble$Symbol != "DOW", ]
lstm_RV_IV_ensemble <- lstm_RV_IV_ensemble[lstm_RV_IV_ensemble$Symbol != "DOW", ]
lstm_RV_ensemble_rolling <- lstm_RV_ensemble_rolling[lstm_RV_ensemble_rolling$Symbol != "DOW", ]
lstm_IV_ensemble_rolling <- lstm_IV_ensemble_rolling[lstm_IV_ensemble_rolling$Symbol != "DOW", ]
lstm_RV_IV_ensemble_rolling <- lstm_RV_IV_ensemble_rolling[lstm_RV_IV_ensemble_rolling$Symbol != "DOW", ]
MDN_ensemble_IV_RV <- MDN_ensemble_IV_RV[MDN_ensemble_IV_RV$Symbol != "DOW", ]
#catboost_RV <- catboost_RV[catboost_RV$Symbol != "DOW", ]
catboost_IV <- catboost_IV[catboost_IV$Symbol != "DOW", ]
catboost_RV_IV <- catboost_RV_IV[catboost_RV_IV$Symbol != "DOW", ]
xgboost_RV <- xgboost_RV[xgboost_RV$Symbol != "DOW", ]
# xgboost_IV <- xgboost_IV[xgboost_IV$Symbol != "DOW", ]
# xgboost_RV_IV <- xgboost_RV_IV[xgboost_RV_IV$Symbol != "DOW", ]
lightgbm_RV <- lightgbm_RV[lightgbm_RV$Symbol != "DOW", ]
lightgbm_IV <- lightgbm_IV[lightgbm_IV$Symbol != "DOW", ]
lightgbm_RV_IV <- lightgbm_RV_IV[lightgbm_RV_IV$Symbol != "DOW", ]
# DB_RV <- DB_RV[DB_RV$Symbol != "DOW", ]
# DB_IV <- DB_IV[DB_IV$Symbol != "DOW", ]
# DB_RV_IV <- DB_RV_IV[DB_RV_IV$Symbol != "DOW", ]
har <- har[har$Symbol != "DOW", ]
harq <- harq[harq$Symbol != "DOW", ]
garch_norm <- garch_norm[garch_norm$Symbol != "DOW", ]
garch_t <- garch_t[garch_t$Symbol != "DOW", ]
rv_garch <- rv_garch[rv_garch$Symbol != "DOW", ]
ar_garch_norm <- ar_garch_norm[ar_garch_norm$Symbol != "DOW", ]
# ar_garch_t <- ar_garch_t[ar_garch_t$Symbol != "DOW", ]
egarch <- egarch[egarch$Symbol != "DOW", ]
#######################################################################################################
 # TEST MODELS
#######################################################################################################


##### Transformers and LSTM ####
alpha_config_lstm_transformer <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "LB_98", "0.025" = "LB_95", "0.05" = "LB_90")
)
es_config_lstm_transformer <- list(
  columns = list("0.01" = "ES_99", "0.025" = "ES_97.5", "0.05" = "ES_95")
)

# Model list
model_list_lstm_transformer <- list(
 "LSTM_RV" = lstm_RV_ensemble,
 "LSTM_IV" = lstm_IV_ensemble,
 "LSTM_RV_IV" = lstm_RV_IV_ensemble,
 "Transformer_RV" = transformer_RV_ensemble,
"Transformer_IV" = transformer_IV_ensemble,
 "Transformer_RV_IV" = transformer_RV_IV_ensemble,
 # "MDN_ensemble_IV_RV" = MDN_ensemble_IV_RV,
  "LSTM_RV_ensemble_rolling" = lstm_RV_ensemble_rolling,
  "LSTM_IV_ensemble_rolling" = lstm_IV_ensemble_rolling,
  "LSTM_RV_IV_ensemble_rolling" = lstm_RV_IV_ensemble_rolling
)


#### GARCH ####
alpha_config_garch <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "LB_98", "0.025" = "LB_95", "0.05" = "LB_90")
)
es_config_garch <- list(
  columns = list("0.01" = "ES_99", "0.025" = "ES_97.5", "0.05" = "ES_95")
)

# Model list
model_list_garch <- list(
  "GARCH" = garch_norm,
  "EGARCH" = egarch
  #"RV_GARCH" = rv_garch,
 #"AR_GARCH" = ar_garch_norm,
  #"AR_GARCH_t" = ar_garch_t
  #"GARCH_t" = garch_t

)

#### BOOSTERS ####
alpha_config_boost <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "Quantile_0.010", "0.025" = "Quantile_0.025", "0.05" = "Quantile_0.050")
)
es_config_boost <- list(
  columns = list("0.01" = "ES_0.010", "0.025" = "ES_0.025", "0.05" = "ES_0.050")
)
# Model list
model_list_boosters <- list(
  #"CatBoost_RV" = catboost_RV,
  "CatBoost_IV" = catboost_IV,
  "CatBoost_RV_IV" = catboost_RV_IV,
  "XGBoost_RV" = xgboost_RV,
  "XGBoost_IV" = xgboost_IV,
  "XGBoost_RV_IV" = xgboost_RV_IV,
  "LightGBM_RV" = lightgbm_RV,
  "LightGBM_IV" = lightgbm_IV,
  "LightGBM_RV_IV" = lightgbm_RV_IV
)


### DB ####
alpha_config_db <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "DB_RV_set_0.01", "0.025" = "DB_RV_set_0.025", "0.05" = "DB_RV_set_0.05")
)
es_config_db <- list(
  columns = list("0.01" = "DB_RV_set_ES_0.01", "0.025" = "DB_RV_set_ES_0.025", "0.05" = "DB_RV_set_ES_0.05")
)
# Model list

model_list_DB <- list(
#  "DB_RV" = DB_RV,
#  "DB_IV" = DB_IV,
#  "DB_RV_IV" = DB_RV_IV
)



#### HAR ####
alpha_config_har <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "LB_98", "0.025" = "LB_95", "0.05" = "LB_90")
)
es_config_har <- list(
  columns = list("0.01" = "ES_99", "0.025" = "ES_97.5", "0.05" = "ES_95")
)

# Model list
model_list_HAR <- list(
  "HAR" = har,
  "HARQ" = harq
)

#####################################################
# Run the backtest for all models and combinations
#####################################################





####################### DEFINE FUNCTION THAT DOES IT ALL #############################################################
run_esr_backtests <- function(all_model_groups, return_data, test_versions = c(2), sig_levels = c(0.05)) {
  all_results <- list()
  
  for (test_version in test_versions) {
    for (sig in sig_levels) {
      cat("Running ESR backtest for test version:", test_version, "and significance level:", sig, "\n")
      all_combined <- data.frame()
      
      for (group in all_model_groups) {
        model_list <- group$models
        alpha_config <- group$alpha_config
        es_config <- group$es_config
        
        for (model_name in names(model_list)) {
          model_data <- model_list[[model_name]]
          symbols <- unique(model_data$Symbol)
          
          for (alpha in alpha_config$levels) {
            pass_count <- 0
            fail_count <- 0
            
            for (sym in symbols) {
              model_sym_data <- model_data[model_data$Symbol == sym, ]
              return_sym_data <- return_data[return_data$Symbol == sym, ]
              
              if (nrow(model_sym_data) != nrow(return_sym_data)) next
              if (anyNA(model_sym_data) || anyNA(return_sym_data)) next
              
              r <- return_sym_data$LogReturn
              q <- model_sym_data[[alpha_config$columns[[as.character(alpha)]]]]
              e <- model_sym_data[[es_config$columns[[as.character(alpha)]]]]
              
              result <- tryCatch({
                esr_backtest(
                  r = r,
                  q = q,
                  e = e,
                  alpha = alpha,
                  version = test_version,
                  cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
                )
              }, error = function(e) return(NULL))
              
              if (!is.null(result)) {
                pval <- result$pvalue_twosided_asymptotic
                if (!is.null(pval)) {
                  if (pval >= sig) pass_count <- pass_count + 1 else fail_count <- fail_count + 1
                }
              }
            }
            
            total <- pass_count + fail_count
            fail_rate <- ifelse(total > 0, fail_count / total, NA)
            
            all_combined <- rbind(all_combined, data.frame(
              Model = model_name,
              Alpha = alpha,
              TestVersion = test_version,
              Significance = sig,
              SymbolsTested = total,
              Passed = pass_count,
              Failed = fail_count,
              FailRate = round(fail_rate * 100, 2)
            ))
          }
        }
      }
      
      key <- paste0("Version_", test_version, "_Sig_", sig)
      all_results[[key]] <- all_combined
    }
  }
  
  return(all_results)
}



############################ Defining all model combinations ##################################################
all_model_groups <- list(
  list(models = model_list_lstm_transformer, alpha_config = alpha_config_lstm_transformer, es_config = es_config_lstm_transformer),
  #list(models = model_list_boosters, alpha_config = alpha_config_boost, es_config = es_config_boost),
  #list(models = model_list_DB, alpha_config = alpha_config_db, es_config = es_config_db),
  #list(models = model_list_HAR, alpha_config = alpha_config_har, es_config = es_config_har),
  list(models = model_list_garch, alpha_config = alpha_config_garch, es_config = es_config_garch)
)
############################## Running to get all result variations ###########################################
esr_results <- run_esr_backtests(
  all_model_groups = all_model_groups,
  return_data = return_data
)

library(knitr)
library(kableExtra)

# Displaying each version, significance combination
for (key in names(esr_results)) {
  cat("\n### Results for", key, "\n")
  df <- esr_results[[key]]
  df$Alpha <- factor(df$Alpha, levels = c(0.01, 0.025, 0.05), labels = c("1%", "2.5%", "5%"))
  print(kable(df, format = "pipe", digits = 2, align = 'c'))
}
