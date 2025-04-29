# %%
# Change working directory to Code folder in repo
# **NB** Update this path to your local repo path
setwd("/Users/tordjes/Github/master/Code")

# Install all necessary packages
install.packages(c("data.table", "gridExtra", "tidyverse", "ggplot2", "hrbrthemes", "esreg", "rugarch", "xts", "parallel", "pbapply"))

library(data.table)

# define what variables we are working with
include_rv = TRUE
include_IV = TRUE

# version is "RV" if we include_rv and "IV" if we include_IV and "RV_IV" if we include both
version = ifelse(include_rv & include_IV, "RV_IV", ifelse(include_rv, "RV", "IV"))
print(version)

# filename based on version
# cd to repo
datapath <- paste0("data/processed_data_DB_", version, ".csv")
print(datapath)
# Load necessary libraries
D <- read.csv(datapath)
setDT(D)

depentant_var= 'Return'

# the independand var set has to change based on the version we are using
if (include_rv & include_IV) {
  # if we include both RV and IV, we need to change the independent variable set
  independant_var_sets <- list(
    set1 = 'feat_0 + feat_1 + feat_2 + feat_3 + feat_4 + feat_5 + feat_6 + feat_7 + feat_8 + feat_9 + feat_10 + feat_11'
  )
} else if (include_rv) {
  # if we only include RV, we need to change the independent variable set
  independant_var_sets <- list(
    set1 = 'feat_0 + feat_1 + feat_2 + feat_3 + feat_4 + feat_5 + feat_6 + feat_7 + feat_8 + feat_9'
  )
} else if (include_IV) {
  # if we only include IV, we need to change the independent variable set
  independant_var_sets <- list(
    set1 = 'feat_0 + feat_1'
  )
}

# change name on set based on the version we are using
if (include_rv & include_IV) {
  independant_var_names_set <- list(
    set1 = 'RV_set + IV_set'
  )
} else if (include_rv) {
  independant_var_names_set <- list(
    set1 = 'RV_set'
  )
} else if (include_IV) {
  independant_var_names_set <- list(
    set1 = 'IV_set'
  )
}

independant_var_names <- unname(unlist(independant_var_names_set))

ES <- c(0.01, 0.025, 0.05, 0.165, 0.835, 0.95, 0.975, 0.99)


# Dimitriadis and Bayer model 
DB = function(dt = D, EW = 1500, ES = ES, nc = 16, run_all_variations = FALSE) {
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
  
  # Convert to data.table
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
    # Check for NAs in y
    if (anyNA(y[1:win])) {
      print("Found NA in y!")
      print(y[1:win])
      stop("Stopping due to NA in y.")
    }
    
    # Check for NAs in x
    if (!is.null(x)) {
      if (anyNA(x[1:win, ])) {
        print("Found NA in x!")
        print(x[1:win, ])
        stop("Stopping due to NA in x.")
      }
    }
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
    if (length(independant_var_sets) >= 2) {
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
    if (length(independant_var_sets) >= 2) {
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
    print(es)
    # Loop over model specifications
    for (m in 1:NM) {
      
      # Dependent variable (the same)
      dep  = all.vars(specs[[m]])[1]
      # Independent variables
      exog = all.vars(specs[[m]])[-1]
      # Select data
      y = dt[[dep]]
      # y     = dt[,..dep];
      x     = dt[,..exog]
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
      
      
      dt[, names(res)[-c(1:2)] := res[,-c(1:2)]]
      
      
    }  
  }
  
  return(dt)
  
}



# RUN THE MODEL
# Run the Dimitriadis and Bayer model
# Initialize a list to store results for all tickers
all_results <- list()

# Get unique assets
tickers <- unique(D$Symbol)

# Loop through each asset (Ticker)
for (ticker in tickers) {
  file_name <- paste0("~/Copy of data/DB_", ticker, ".csv")
  print(ticker)
  # Filter dataset for the current asset
  D_ticker <- D[Symbol == ticker]
  
  # Run the DB model on the subset
  results <- DB(dt = D_ticker, 
                EW = 1500, 
                ES = ES, 
                nc = 8, 
                run_all_variations = FALSE) 
  
  # Add the Ticker column to the results
  results[, Symbol := ticker]
  
  # Store results in the list
  all_results[[ticker]] <- results
}

# Combine all results into a single data.table
final_results <- rbindlist(all_results)

# write the final results to a csv file:
final_file_path = paste0("predictions/DB_", version, ".csv")
write.csv(final_results, final_file_path)
