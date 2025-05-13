# %%
# install.packages("esback")  # Only once
library(esback)
library(dplyr)

# UPDATE BASEPATH TO YOUR FILE PATH
base_path_return_data <- "C:///Users/tordjes/Github/master/Code/data/dow_jones/processed_data"
base_path_predictions <- "C:///Users/tordjes/Github/master/Code/predictions"

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
# %%

# non-rolling
transformer_RV_ensemble <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vrv_test_ensemble.csv"))
transformer_IV_ensemble <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vivol_test_ensemble.csv"))
transformer_RV_IV_ensemble <- read.csv(file.path(base_path_predictions, "transformer_mdn_predictions_stocks_vrv-and-ivol_test_ensemble.csv"))

# rolling
transformer_RV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_rvol_test_expanding.csv"))
transformer_IV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_ivol_test_expanding.csv"))
transformer_RV_IV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "transformer_mdn_ensemble_rvol-ivol_test_expanding.csv"))

############ LSTM ###################
# %%

# non-rolling
# lstm_RV_ensemble <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vrv-final_ensemble.csv"))
# lstm_IV_ensemble <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vivol-final_ensemble.csv"))
# lstm_RV_IV_ensemble <- read.csv(file.path(base_path_predictions, "lstm_mdn_predictions_stocks_vrv-and-ivol-final_ensemble.csv"))

# rolling
lstm_RV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vrv-final-rolling_test.csv"))
lstm_IV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vivol-final-rolling_test.csv"))
lstm_RV_IV_ensemble_rolling <- read.csv(file.path(base_path_predictions, "lstm_mdn_ensemble_stocks_vrv-and-ivol-final-rolling_test.csv"))

############## ENSEMBLE COMBINATIONS ###############
# %%
MDN_ensemble_RV <- read.csv(file.path(base_path_predictions, "mdn_ensemble_rv_test_expanding.csv"))
MDN_ensemble_IV <- read.csv(file.path(base_path_predictions, "mdn_ensemble_iv_test_expanding.csv"))
MDN_ensemble_IV_RV <- read.csv(file.path(base_path_predictions, "mdn_ensemble_rv-iv_test_expanding.csv"))

########## GARCH MODELS ###########
# %%
garch_norm <- read.csv(file.path(base_path_predictions, "GARCH_preds_enriched.csv"))
garch_t <- read.csv(file.path(base_path_predictions, "garch_predictions_student_t.csv"))
garch_skew_t <- read.csv(file.path(base_path_predictions, "garch_predictions_skewed_t.csv"))
rv_garch <- read.csv(file.path(base_path_predictions, "realized_garch_forecast_std.csv"))
ar_garch_norm <- read.csv(file.path(base_path_predictions, "predictions_AR(1)-GARCH(1,1)-normal.csv"))
ar_garch_t <- read.csv(file.path(base_path_predictions, "predictions_AR(1)-GARCH(1,1)-t.csv")) # Not included because file name is missing
egarch <- read.csv(file.path(base_path_predictions, "EGARCH_preds_enriched.csv"))


# filter only for test set, from test_set_start_date to 2024-03-28
garch_norm <- garch_norm[garch_norm$Date >= test_set_start_date & garch_norm$Date <= "2024-03-28", ]
garch_t <- garch_t[garch_t$Date >= test_set_start_date & garch_t$Date <= "2024-03-28", ]
garch_skew_t <- garch_skew_t[garch_skew_t$Date >= test_set_start_date & garch_skew_t$Date <= "2024-03-28", ]
rv_garch <- rv_garch[rv_garch$Date >= test_set_start_date & rv_garch$Date <= "2024-03-28", ]
ar_garch_norm <- ar_garch_norm[ar_garch_norm$Date >= test_set_start_date & ar_garch_norm$Date <= "2024-03-28", ]
ar_garch_t <- ar_garch_t[ar_garch_t$Date >= test_set_start_date & ar_garch_t$Date <= "2024-03-28", ]
egarch <- egarch[egarch$Date >= test_set_start_date & egarch$Date <= "2024-03-28", ]

# remove .O at the end of the Symbol for the garch models
garch_norm$Symbol <- gsub("\\.O$", "", garch_norm$Symbol)
garch_t$Symbol <- gsub("\\.O$", "", garch_t$Symbol)
garch_skew_t$Symbol <- gsub("\\.O$", "", garch_skew_t$Symbol)
rv_garch$Symbol <- gsub("\\.O$", "", rv_garch$Symbol)
ar_garch_norm$Symbol <- gsub("\\.O$", "", ar_garch_norm$Symbol)
ar_garch_t$Symbol <- gsub("\\.O$", "", ar_garch_t$Symbol)
egarch$Symbol <- gsub("\\.O$", "", egarch$Symbol)


######### HAR ##############
# %%
har <- read.csv(file.path(base_path_predictions, "HAR_python.csv"))
harq <- read.csv(file.path(base_path_predictions, "HARQ_python.csv"))
har_qreq <- read.csv(file.path(base_path_predictions, "HAR_qreg_test.csv"))
har_ivol_qreq <- read.csv(file.path(base_path_predictions, "HAR_IVOL_qreg_test.csv"))

########### BOOSTERS ###########
# %%
catboost_RV <- read.csv(file.path(base_path_predictions, "CatBoost_RV_4y.csv"))
catboost_IV <- read.csv(file.path(base_path_predictions, "CatBoost_IV_4y.csv"))
catboost_RV_IV <- read.csv(file.path(base_path_predictions, "CatBoost_RV_IV_4y.csv"))

xgboost_RV <- read.csv(file.path(base_path_predictions, "XGBoost_RV_4y.csv"))
xgboost_IV <- read.csv(file.path(base_path_predictions, "XGBoost_IV_4y.csv"))
xgboost_RV_IV <- read.csv(file.path(base_path_predictions, "XGBoost_RV_IV_4y.csv"))

lightgbm_RV <- read.csv(file.path(base_path_predictions, "LightGBM_RV_4y.csv"))
lightgbm_IV <- read.csv(file.path(base_path_predictions, "LightGBM_IV_4y.csv"))
lightgbm_RV_IV <- read.csv(file.path(base_path_predictions, "LightGBM_RV_IV_4y.csv"))

########## DB ###########
# %%
DB_RV <- read.csv(file.path(base_path_predictions, "DB_RV.csv"))
DB_IV <- read.csv(file.path(base_path_predictions, "DB_IV.csv"))
DB_RV_IV <- read.csv(file.path(base_path_predictions, "DB_RV_IV.csv"), check.names = FALSE)

# Filter away dates before test_set_start_date
DB_RV <- DB_RV[DB_RV$Date >= test_set_start_date, ]
DB_IV <- DB_IV[DB_IV$Date >= test_set_start_date, ]
DB_RV_IV <- DB_RV_IV[DB_RV_IV$Date >= test_set_start_date, ]


# %%
# remove ticker "DOW" from all models if it exists
transformer_RV_ensemble <- transformer_RV_ensemble[transformer_RV_ensemble$Symbol != "DOW", ]
transformer_IV_ensemble <- transformer_IV_ensemble[transformer_IV_ensemble$Symbol != "DOW", ]
transformer_RV_IV_ensemble <- transformer_RV_IV_ensemble[transformer_RV_IV_ensemble$Symbol != "DOW", ]
transformer_RV_ensemble_rolling <- transformer_RV_ensemble_rolling[transformer_RV_ensemble_rolling$Symbol != "DOW", ]
transformer_IV_ensemble_rolling <- transformer_IV_ensemble_rolling[transformer_IV_ensemble_rolling$Symbol != "DOW", ]
transformer_RV_IV_ensemble_rolling <- transformer_RV_IV_ensemble_rolling[transformer_RV_IV_ensemble_rolling$Symbol != "DOW", ]
# lstm_RV_ensemble <- lstm_RV_ensemble[lstm_RV_ensemble$Symbol != "DOW", ]
# lstm_IV_ensemble <- lstm_IV_ensemble[lstm_IV_ensemble$Symbol != "DOW", ]
# lstm_RV_IV_ensemble <- lstm_RV_IV_ensemble[lstm_RV_IV_ensemble$Symbol != "DOW", ]
lstm_RV_ensemble_rolling <- lstm_RV_ensemble_rolling[lstm_RV_ensemble_rolling$Symbol != "DOW", ]
lstm_IV_ensemble_rolling <- lstm_IV_ensemble_rolling[lstm_IV_ensemble_rolling$Symbol != "DOW", ]
lstm_RV_IV_ensemble_rolling <- lstm_RV_IV_ensemble_rolling[lstm_RV_IV_ensemble_rolling$Symbol != "DOW", ]
MDN_ensemble_RV <- MDN_ensemble_RV[MDN_ensemble_RV$Symbol != "DOW", ]
MDN_ensemble_IV <- MDN_ensemble_IV[MDN_ensemble_IV$Symbol != "DOW", ]
MDN_ensemble_IV_RV <- MDN_ensemble_IV_RV[MDN_ensemble_IV_RV$Symbol != "DOW", ]
catboost_RV <- catboost_RV[catboost_RV$Symbol != "DOW", ]
catboost_IV <- catboost_IV[catboost_IV$Symbol != "DOW", ]
catboost_RV_IV <- catboost_RV_IV[catboost_RV_IV$Symbol != "DOW", ]
xgboost_RV <- xgboost_RV[xgboost_RV$Symbol != "DOW", ]
xgboost_IV <- xgboost_IV[xgboost_IV$Symbol != "DOW", ]
xgboost_RV_IV <- xgboost_RV_IV[xgboost_RV_IV$Symbol != "DOW", ]
lightgbm_RV <- lightgbm_RV[lightgbm_RV$Symbol != "DOW", ]
lightgbm_IV <- lightgbm_IV[lightgbm_IV$Symbol != "DOW", ]
lightgbm_RV_IV <- lightgbm_RV_IV[lightgbm_RV_IV$Symbol != "DOW", ]
DB_RV <- DB_RV[DB_RV$Symbol != "DOW", ]
DB_IV <- DB_IV[DB_IV$Symbol != "DOW", ]
DB_RV_IV <- DB_RV_IV[DB_RV_IV$Symbol != "DOW", ]
har <- har[har$Symbol != "DOW", ]
harq <- harq[harq$Symbol != "DOW", ]
har_qreq <- har_qreq[har_qreq$Symbol != "DOW", ]
har_ivol_qreq <- har_ivol_qreq[har_ivol_qreq$Symbol != "DOW", ]
garch_norm <- garch_norm[garch_norm$Symbol != "DOW", ]
garch_t <- garch_t[garch_t$Symbol != "DOW", ]
garch_skew_t <- garch_skew_t[garch_skew_t$Symbol != "DOW", ]
rv_garch <- rv_garch[rv_garch$Symbol != "DOW", ]
ar_garch_norm <- ar_garch_norm[ar_garch_norm$Symbol != "DOW", ]
ar_garch_t <- ar_garch_t[ar_garch_t$Symbol != "DOW", ]
egarch <- egarch[egarch$Symbol != "DOW", ]
#######################################################################################################
# TEST MODELS
#######################################################################################################


##### Transformers and LSTM ####
# %%
alpha_config_lstm_transformer <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "LB_98", "0.025" = "LB_95", "0.05" = "LB_90")
)
es_config_lstm_transformer <- list(
  columns = list("0.01" = "ES_99", "0.025" = "ES_97.5", "0.05" = "ES_95")
)

# Model list
model_list_lstm_transformer <- list(
  "LSTM_RV_ensemble_rolling" = lstm_RV_ensemble_rolling,
  "LSTM_IV_ensemble_rolling" = lstm_IV_ensemble_rolling,
  "LSTM_RV_IV_ensemble_rolling" = lstm_RV_IV_ensemble_rolling,
  "Transformer_RV_ensemble_rolling" = transformer_RV_ensemble_rolling,
  "Transformer_IV_ensemble_rolling" = transformer_IV_ensemble_rolling,
  "Transformer_RV_IV_ensemble_rolling" = transformer_RV_IV_ensemble_rolling,
  "MDN_ensemble_RV" = MDN_ensemble_RV,
  "MDN_ensemble_IV" = MDN_ensemble_IV,
  "MDN_ensemble_IV_RV" = MDN_ensemble_IV_RV
)


#### GARCH ####
# %%
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
  "EGARCH" = egarch,
  "RV_GARCH" = rv_garch,
  "AR_GARCH" = ar_garch_norm,
  "AR_GARCH_t" = ar_garch_t,
  "GARCH_t" = garch_t,
  "GARCH_skew_t" = garch_skew_t
)

#### BOOSTERS ####
# %%
alpha_config_boost <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "Quantile_0.010", "0.025" = "Quantile_0.025", "0.05" = "Quantile_0.050")
)
es_config_boost <- list(
  columns = list("0.01" = "ES_0.010", "0.025" = "ES_0.025", "0.05" = "ES_0.050")
)
# Model list
model_list_boosters <- list(
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


### DB ####
# %%
alpha_config_db_rv <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "DB_RV_set_0.01", "0.025" = "DB_RV_set_0.025", "0.05" = "DB_RV_set_0.05")
)
alpha_config_db_iv <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "DB_IV_set_0.01", "0.025" = "DB_IV_set_0.025", "0.05" = "DB_IV_set_0.05")
)
alpha_config_db_rv_iv <- list(
  levels = c(0.01, 0.025, 0.05),
  columns = list("0.01" = "DB_RV_set + IV_set_0.01", "0.025" = "DB_RV_set + IV_set_0.025", "0.05" = "DB_RV_set + IV_set_0.05")
)
es_config_db_rv <- list(
  columns = list("0.01" = "DB_RV_set_ES_0.01", "0.025" = "DB_RV_set_ES_0.025", "0.05" = "DB_RV_set_ES_0.05")
)
es_config_db_iv <- list(
  columns = list("0.01" = "DB_IV_set_ES_0.01", "0.025" = "DB_IV_set_ES_0.025", "0.05" = "DB_IV_set_ES_0.05")
)
es_config_db_rv_iv <- list(
  columns = list("0.01" = "DB_RV_set + IV_set_ES_0.01", "0.025" = "DB_RV_set + IV_set_ES_0.025", "0.05" = "DB_RV_set + IV_set_ES_0.05")
)

# Model lists
model_list_DB_RV <- list(
  "DB_RV" = DB_RV
)
model_list_DB_IV <- list(
  "DB_IV" = DB_IV
)
model_list_DB_RV_IV <- list(
  "DB_RV_IV" = DB_RV_IV
)



#### HAR ####
# %%
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
  "HARQ" = harq,
  "HAR_QREG" = har_qreq,
  "HAR_IVOL_QREG" = har_ivol_qreq
)

#####################################################
# Run the backtest for all models and combinations
#####################################################

# %%
names(transformer_RV_ensemble_rolling)
head(transformer_RV_ensemble_rolling[, grep("^ES", names(transformer_RV_ensemble_rolling))])


####################### DEFINE FUNCTION THAT DOES IT ALL #############################################################
# %%
run_esr_backtests <- function(all_model_groups, return_data, test_versions = c(1), sig_levels = c(0.05)) {
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
          # Print model name
          cat("  Model:", model_name, "\n")


          for (alpha in alpha_config$levels) {
            cat("    Alpha level:", alpha, "\n")
            pass_count <- 0
            fail_count <- 0

            alpha_col <- alpha_config$columns[[as.character(alpha)]]
            es_col <- es_config$columns[[as.character(alpha)]]

            for (sym in symbols) {
              # subset and align by Date
              returns <- return_data %>%
                filter(Symbol == sym) %>%
                select(Date, LogReturn) %>%
                arrange(Date)

              preds <- model_data %>%
                filter(Symbol == sym) %>%
                select(Date, all_of(c(alpha_col, es_col))) %>%
                arrange(Date)

              # do an inner???join to line up dates exactly
              joint <- inner_join(returns, preds, by = "Date")
              if (nrow(joint) == 0) next # no overlap

              # now extract r, q, e
              r <- joint$LogReturn
              q <- joint[[alpha_col]]
              e <- joint[[es_col]]

              result <- tryCatch(
                {
                  esr_backtest(
                    r = r,
                    q = q,
                    e = e,
                    alpha = alpha,
                    version = test_version,
                    cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
                  )
                },
                error = function(e) {
                  return(NULL)
                }
              )

              if (!is.null(result)) {
                pval <- result$pvalue_twosided_asymptotic
                if (!is.null(pval)) {
                  if (pval >= sig) pass_count <- pass_count + 1 else fail_count <- fail_count + 1
                  # Print the result for each symbol
                  cat("      Symbol:", sym, "p-value:", round(pval, 3), ifelse(pval >= sig, "[PASS]", "[FAIL]"), "\n")
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
# %%
all_model_groups <- list(
  list(models = model_list_lstm_transformer, alpha_config = alpha_config_lstm_transformer, es_config = es_config_lstm_transformer),
  list(models = model_list_boosters, alpha_config = alpha_config_boost, es_config = es_config_boost),
  list(models = model_list_DB_RV, alpha_config = alpha_config_db_rv, es_config = es_config_db_rv),
  list(models = model_list_DB_IV, alpha_config = alpha_config_db_iv, es_config = es_config_db_iv),
  list(models = model_list_DB_RV_IV, alpha_config = alpha_config_db_rv_iv, es_config = es_config_db_rv_iv),
  list(models = model_list_HAR, alpha_config = alpha_config_har, es_config = es_config_har),
  list(models = model_list_garch, alpha_config = alpha_config_garch, es_config = es_config_garch)
)
############################## Running to get all result variations ###########################################
# %%
esr_results <- run_esr_backtests(
  all_model_groups = all_model_groups,
  return_data = return_data
)

# %%
library(knitr)
library(kableExtra)

# %%
# Displaying each version, significance combination
for (key in names(esr_results)) {
  cat("\n### Results for", key, "\n")
  df <- esr_results[[key]]
  df$Alpha <- factor(df$Alpha, levels = c(0.01, 0.025, 0.05), labels = c("1%", "2.5%", "5%"))
  print(kable(df, format = "pipe", digits = 2, align = "c"))
}


# %%
# Print as LaTeX table
for (key in names(esr_results)) {
  cat("\n\\begin{table}[H]\n")
  cat("\\centering\n")
  cat("\\caption{ESR p-values for", key, "}\n")
  cat("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
  cat("\\hline\n")
  cat("Model & Alpha & TestVersion & Significance & SymbolsTested & Passed & Failed \\\\\n")
  cat("\\hline\n")

  df <- esr_results[[key]]
  df$Alpha <- factor(df$Alpha, levels = c(0.01, 0.025, 0.05), labels = c("1%", "2.5%", "5%"))

  for (i in 1:nrow(df)) {
    row <- df[i, ]
    cat(paste(row$Model, "&", row$Alpha, "&", row$TestVersion, "&", row$Significance, "&", row$SymbolsTested, "&", row$Passed, "&", row$Failed, "\\\\\n"))
    cat("\\hline\n")
  }

  cat("\\end{tabular}\n")
  cat("\\end{table}\n")
}



# IGNORE THIS
###########################################################
# isolate one model to see what happens
# pick your model & return data
model_df <- transformer_RV_IV_ensemble_rolling
return_df <- return_data

symbols <- unique(model_df$Symbol)
alphas <- alpha_config_lstm_transformer$levels

# prepare a list to collect everything
results_list <- list()

for (alpha in alphas) {
  alpha_col <- alpha_config_lstm_transformer$columns[[as.character(alpha)]]
  es_col <- es_config_lstm_transformer$columns[[as.character(alpha)]]

  cat("\n=== ?? =", alpha, "(", alpha_col, "/", es_col, ") ===\n")

  for (sym in symbols) {
    cat("  ???", sym, "??? ")

    # grab + align
    returns <- return_df %>%
      filter(Symbol == sym) %>%
      select(Date, LogReturn) %>%
      arrange(Date)

    preds <- model_df %>%
      filter(Symbol == sym) %>%
      select(Date, all_of(c(alpha_col, es_col))) %>%
      arrange(Date)

    joint <- inner_join(returns, preds, by = "Date")
    if (nrow(joint) == 0) {
      cat("no dates ??? skip\n")
      next
    }

    # run ESR
    res <- tryCatch(
      esr_backtest(
        r = joint$LogReturn,
        q = joint[[alpha_col]],
        e = joint[[es_col]],
        alpha = alpha,
        version = 1,
        cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE)
      ),
      error = function(e) NULL
    )

    if (is.null(res) || is.null(res$pvalue_twosided_asymptotic)) {
      cat("error/NA p-val\n")
      next
    }

    pval <- res$pvalue_twosided_asymptotic
    passed <- pval >= 0.05
    cat(
      "p-val=", round(pval, 3),
      ifelse(passed, "[PASS]\n", "[FAIL]\n")
    )

    # store
    results_list[[length(results_list) + 1]] <- data.frame(
      Alpha = alpha,
      Symbol = sym,
      p_value = pval,
      Passed = passed,
      stringsAsFactors = FALSE
    )
  }
}

# combine & display
library(knitr)
all_sym_results <- do.call(rbind, results_list)
kable(all_sym_results, digits = 3, caption = "ESR p-values")
