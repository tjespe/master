# %%
# Load required libraries
library(data.table)
library(openxlsx)

# %%
# Load data
D <- fread("D.csv") # input the correct filename here
# Ensure Date is in the correct format
D[, Date := as.Date(Date)]
# define the Eexpected Shortfall levels
ES <- c(0.01, 0.025, 0.05, 0.95, 0.975, 0.99)


# NOT QUITE SURE WHAT THIS IS DOING, think it is easier to just specify one set of independent variables
# Define independent variable sets based on your data 
independant_var_sets <- list(
  set1 = 'Feature_A + Feature_B + Feature_C',  # Replace with actual column names
  set2 = 'Feature_D + Feature_E + Feature_F',
  set3 = 'Feature_G + Feature_H + Feature_I'
)

# define meaningul names for the independent variable sets
independant_var_names_set <- list(
  set1 = 'FeatureSet1',
  set2 = 'FeatureSet2',
  set3 = 'FeatureSet3'
)

# Define the dependent variable
depentant_var= 'Return'

# %% 
# Define the function to run the Dimitriadis and Bayer model
# Dimitriadis and Bayer model 
DB = function(dt = D, EW = 1500, ES = ES, nc = 16,run_all_variations = FALSE, writefile="writefile") {
  
  library(gridExtra)
  library(tidyverse);
  library(ggplot2);
  library(hrbrthemes);
  library(esreg);
  library(rugarch);
  library(xts);
  library(data.table);
  library(parallel);
  library(pbapply)
  
  # Conert to data.table
  # setDT(dt)
  
  # --==========================================================================================
  # t_forecast_QES_DB robi Dimitriadis, T., & Bayer, S. (2019). A joint quantile and expected shortfall regression framework.
  # modeluje sa VaR a ES y pomocou *lagovanych* x-iek
  t_forecast_QES_DB = function(y,x,alpha) {
    if (!is.null(x)) {
      if (!('matrix' %in% class(x))) x=as.matrix(x)
      if (length(y)!=nrow(x)) stop('t_forecast_QES_DB: invalid input dimensions')
    }
    chng = F
    if (alpha > 0.5) {
      y = y*-1
      x = x*-1
      alpha = 1-alpha
      chng = T
    }
    win=length(y)-1
    if (is.null(x)) fit = esreg::esreg(y[1:win] ~ 1, alpha=alpha) else fit = esreg::esreg(y[1:win] ~ x[1:win,], alpha=alpha)
    q = as.numeric(c(1, x[length(y),]) %*% fit$coefficients_q)
    e = as.numeric(c(1, x[length(y),]) %*% fit$coefficients_e)
    if (chng) {
      return(data.frame(y=y[length(y)],q=q,e=e)*-1)
    } else {
      return(data.frame(y=y[length(y)],q=q,e=e))
    }
    
  }
  
  # --==========================================================================================
  # roll forecasts - DB
  t_roll_QES_DB = function(y,x=NULL,dates,alpha=0.1,windowSize=1500, parallel=T, nc=16) {
    df=data.frame(date=dates,y=y); if (!is.null(x)) df=data.frame(df,x); i=windowSize+1
    
    #detectCores(logical=F)
    cl=NULL; if (parallel) { cl=makePSOCKcluster(nc,outfile=''); clusterExport(cl,c('t_forecast_QES_DB','df','alpha','windowSize'),envir=environment()) }
    res=rbindlist(pblapply((windowSize+1):(length(dates)), function(i) {
      yy=df[(i-windowSize):i,][[2]]
      xx=data.frame(df[(i-windowSize):i,-c(1,2)]); if (ncol(xx)==0) xx=NULL
      t_forecast_QES_DB(y=yy,x=xx,alpha=alpha)
    }, cl=cl))
    if (!is.null(cl)) stopCluster(cl); cl=NULL
    tmp=data.frame(date=dates[(windowSize+1):length(dates)]); 
    tmp$model=paste0('DB_',alpha); 
    tmp$y=res$y;
    tmp$q=res$q;
    tmp$e=res$e
    tmp
  }
  
  # Specifications
  if (run_all_variations){
    
    # Initialize an empty list to store your specs
    specs <- list()
    model_names <- c()
    counter <- 1
    
    # Get the names of the sets
    set_names <- names(independant_var_sets)
    
    # Generate the specs for each set on its own
    for (i in seq_along(independant_var_sets)) {
      current_vars <- independant_var_sets[[i]]
      specs[[paste0("DB.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
      
      # Generate the model name
      model_name <- paste0("DB_", independant_var_names[[i]])
      model_names <- c(model_names, model_name)
      counter <- counter + 1
    }
    
    # Generate the specs for each combination of sets
    for (n in 2:length(independant_var_sets)) {
      combinations <- combn(set_names, n, simplify = FALSE)
      
      for (combo in combinations) {
        current_vars <- paste(unlist(independant_var_sets[combo]), collapse = " + ")
        specs[[paste0("DB.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
        
        # Generate the model name
        model_name <- paste0("DB_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        
        
        model_names <- c(model_names, model_name)
        
        counter <- counter + 1
      }
    }
    # Assign the generated names to the specs
    names(specs) <- model_names
  }
  
  else{
    # Initialize an empty list to store your specs
    specs <- list()
    model_names <- c()
    counter <- 1
    
    # Get the names of the sets
    set_names <- names(independant_var_sets)
    
    # Generate the specs for each set on its own
    for (i in seq_along(independant_var_sets)) {
      current_vars <- independant_var_sets[[i]]
      specs[[paste0("DB.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
      
      # Generate the model name
      model_name <- paste0("DB_", independant_var_names[[i]])
      model_names <- c(model_names, model_name)
      counter <- counter + 1
      break
    }
    
    # Generate the specs for each combination of sets
    for (n in 2:length(independant_var_sets)) {
      combinations <- combn(set_names, n, simplify = FALSE)
      
      
      for (combo in combinations) {
        
        if (!"set1" %in% combo) {
          next  # Skip this combination if "set1" is not included
        }
        current_vars <- paste(unlist(independant_var_sets[combo]), collapse = " + ")
        specs[[paste0("DB.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
        
        # Generate the model name
        model_name <- paste0("DB_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        model_names <- c(model_names, model_name)
        counter <- counter + 1
      }
    }
    
    # Assign the generated names to the specs
    names(specs) <- model_names
    
    
  }
  
  
  
  # Number of specifications
  NM = length(specs);
  
  # Number of ES of interest
  NES = length(ES)
  
  # Number of observations
  TT = dim(dt)[1]
  
  # Loop over Expected Shortfalls
  for (w in 1:NES) {
    es = ES[w]
    
    # Loop over model specifications
    for (m in 1:NM) {
      
      # Dependent variable (the same)
      dep  = all.vars(specs[[m]])[1]
      # Independent variables
      exog = all.vars(specs[[m]])[-1]
      # Select data
      y     = dt[,which(names(dt) == dep)]
      # y     = dt[,..dep];
      x     = dt[,exog]
      # x     = dt[,..exog;
      dates = dt$Date;
      
      # Store results here
      res = data.frame(Date=dates,Realized=y,q=NA,e=NA)
      names(res) = c('Date','Realized',paste(names(specs)[m],'_',es,sep=''),paste(names(specs)[m],'_ES_',es,sep=''))
      
      # --=================================================================
      # spocitame rolling forecasts
      
      A = Sys.time()
      tmp = t_roll_QES_DB(y=y, x=x, dates=dates, alpha=es, windowSize=EW, parallel=T, nc = nc) 
      print(Sys.time()-A)
      
      
      
      res[which(dt$Date %in% tmp$date),-c(1:2)] = tmp[,c('q','e')]
      print(tail(res))
      
      res[1:EW,-c(1:2)]                        = NA
      for (i in (EW+2):TT) {
        # Test lower bound
        cond = which(res[i,-c(1:2)] < res[i-1,-c(1:2)] - 6*sd(dt[(i-EW+1):i,all.vars(specs[[m]])[1]],na.rm=T))
        if (length(cond)>0) {
          res[i,-c(1:2)] = res[i-1,-c(1:2)]
        }
        # Test upper bound
        cond = which(res[i,-c(1:2)] > res[i-1,-c(1:2)] + 6*sd(dt[(i-EW+1):i,all.vars(specs[[m]])[1]],na.rm=T))
        if (length(cond)>0) {
          res[i,-c(1:2)] = res[i-1,-c(1:2)]
        }
        # Check positiveness/negativeness & Low or High Var should not be near Median
        if (es < 0.5) {
          # positiveness of low quantiles (not good)
          if (res[i,3] > 0) res[i,3] = 0 - 0.0001
          # improbable values
          if (res[i,3] < quantile(res[(i-EW+1):(i-1),3],p=0.65,na.rm=T) & res[i,3] > quantile(res[(i-EW+1):(i-1),3],p=0.35,na.rm=T)) res[i,3] = res[i-1,3]
          # What if ES > VaR?
          if (res[i,4] > res[i,3]) res[i,c(3,4)] = res[i-1,c(3,4)]
        } else {
          # positiveness of low quantiles (not good)
          if (res[i,3] < 0) res[i,3] = 0 + 0.0001
          # improbable values
          if (res[i,3] < quantile(res[(i-EW+1):(i-1),3],p=0.65,na.rm=T) & res[i,3] > quantile(res[(i-EW+1):(i-1),3],p=0.35,na.rm=T)) res[i,3] = res[i-1,3]
          # What if ES < VaR?
          if (res[i,4] < res[i,3]) res[i,c(3,4)] = res[i-1,c(3,4)]
        }
        
      }
      
      
      dt = data.frame(dt,res[,-c(1:2)])
      
      
    }
    
    
    # Specify the path where you want to save the Excel file
    #filePath_DB <- "/Users/hansmagnusutne/Library/Mobile Documents/com~apple~CloudDocs/Documents/Dokumenter – Hanss MacBook Air/NTNU Dokumenter/Indøk 5. klasse/Forskningsassistent/testFinalDB_2.xlsx"
    
    # Write the final dt to an Excel file
    #write.xlsx(as.data.frame(dt), file = writefile)
    
    
  }
  
  
  
  return(dt)
  
}

# %%
# Run the Dimitriadis and Bayer model
# Initialize a list to store results for all tickers
all_results <- list()

# Get unique assets
tickers <- unique(D$Ticker)

# Loop through each asset (Ticker)
for (ticker in tickers) {
  
  # Filter dataset for the current asset
  D_ticker <- D[Ticker == ticker]
  
  # Run the DB model on the subset
  results <- DB(dt = D_ticker, 
                EW = 1500, 
                ES = ES, 
                nc = 8, 
                run_all_variations = FALSE, 
                writefile=NULL) 
  
  # Add the Ticker column to the results
  results[, Ticker := ticker]
  
  # Store results in the list
  all_results[[ticker]] <- results
}

# Combine all results into a single data.table
final_results <- rbindlist(all_results)

# Save the final results to a csv file
fwrite(final_results, "../../../predictions/Benchmark_DB_ES.csv") # change filename
