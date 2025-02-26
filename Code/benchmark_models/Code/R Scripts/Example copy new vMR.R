################################
# rm(list=ls())
################################
# For example purposes - this is a sub-sample
library(openxlsx)
#DT <- read.xlsx("/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/old_cleaned_data2.xlsx")

DT <- read.xlsx("/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/BRENT Input Eviews.xlsx")
print(colnames(DT))

DT$Date <- as.Date(DT$Date)
sum(is.na(DT$Date)) # This should now be 0

library(data.table)

#depvar_in_file = "Change"
depvar_in_file = "Brent.Returns"
setnames(DT, old = depvar_in_file, new = "Change")

depentant_var= 'Change'
independant_var_sets <- list(
  set1 = 'IV',
  set2 = 'IS'
  ,set3 = 'IK'
  ,set4 = 'Slope'
)
independant_var_names_set <- list(
  set1 = 'IV',
  set2 = 'IS'
  ,set3 = 'IK'
  ,set4 = 'Slope'
)

independant_var_names <- unname(unlist(independant_var_names_set))


#################################################
# Which Are the Expected Shortfalls are of interest?
ES = c(0.010, 0.025, 0.050, 0.950, 0.975, 0.990)
# nc - number of cores
# EW - estimation widnow size
# p  - number of quantiles to calculate (for approximating ES if needed or of interest)
# distrib - assumption about the underlying distribution - see ?ugarchspec

#################################################
single_var <- 'IV_IS_IK'
#single_var <- 'IV'
# Perform traditional GARCH based analysis -> predict multiple quantiles and 'integrate' (approximation) into ES or calculated ES directly from the return distribution
A = Sys.time()
# Then, you can run your function, making sure to use 'DT' instead of 'dt':
#50 minutes ++
D <- ArgarchVaR(dt = DT, EW = 1500, p = 5, ES = ES, distrib = 'sstd', nc = 24,run_all_variations = FALSE, run_single = TRUE)
Sys.time()-A
D = D$data
write.xlsx(D, file = "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/Agarch_BRENT_Test.xlsx")

#################################################
# Perform quantile HAR based analysis -> based on the famous Haugom et al., (2016) paper with HAR-Q models and ES is found via integration again
A = Sys.time()
D = QuantileVaR(dt = DT, EW = 1500 , p = 5, ES = ES, nc = 24, run_all_variations = FALSE)
Sys.time()-A
D = D$data
write.xlsx(D, file = "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/QuantileHAR_WTI.xlsx")

#################################################
# Dimitriadis and Bayer model - this will take a while longer (47minutes on 16 cores)
A = Sys.time()
filePath_DB <- "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/DB_WTI_FINAL.xlsx"
D = DB(dt = DT, EW = 1500, ES = ES, nc = 24,run_all_variations = FALSE, writefile = filePath_DB)
Sys.time()-A
D = D$data
#write.xlsx(D, file = "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/testFinalDB.xlsx")

#################################################
plot.ts(D$R.H1,lwd=1.5)
# 5% ES from DB model - no exogenous just daily, weekly, monthly vola as independent vars
lines(D$DB_ES_0.01,col='red')
# 5% ES from QHAR models - just daily, weekly monthly vola
lines(D$QHAR_ES_0.05,col='blue')
# 5% ES from GARCH with Skewed Student t distribution
lines(D$gNO_sstd_ES_0.05,col='purple')
# 5% VaR from DB model
lines(D$DB_0.01,col='darkred',lty=2,lwd=1.5)


#write.xlsx(D, file = "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter ??? Hanss MacBook Air/NTNU Dokumenter/Ind??k 5. klasse/Forskningsassistent/testing.xlsx")
