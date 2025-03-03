depentant_var= 'Change'

independant_var_sets <- list(
  set1 = 'D_1_Put_50 + W_1_Put_50 + M_1_Put_50',
  set2 = 'D_1_RR_25 + W_1_RR_25 + M_1_RR_25',
  set3 = 'D_1_Call_50 + W_1_Call_50 + M_1_Call_50'
)
independant_var_names_set <- list(
  set1 = 'IV',
  set2 = 'RR',
  set3 = 'VRP'
)

independant_var_names <- unname(unlist(independant_var_names_set))

# Function to convert string sets to vectors of variable names
# convert_to_vector <- function(set_string) {
#   # Split the string by " + " and return as a vector
#   strsplit(set_string, " \\+ ")[[1]]
# }

# Apply the conversion function to each set in the list
#converted_var_sets <- lapply(independant_var_sets, convert_to_vector)


print(independant_var_sets)


ArgarchVaR = function(dt = A, EW = 1500, p = 5, ES = ES, 
                      distrib = 'sstd', nc = 24,run_all_variations = FALSE, run_single = FALSE) {
  library(rugarch)
  library(xts)
  library(foreach)
  library(doParallel)
  
  # Function to convert string sets to vectors of variable names
  convert_to_vector <- function(set_string) {
    # Split the string by " + " and return as a vector
    strsplit(set_string, " \\+ ")[[1]]
  }
  
  # Number of observations
  TT = dim(dt)[1]
  # Number of out-of-sample observations
  TF = TT - EW
  # Number of ES quantiles
  NES = length(ES)
  # Identify, which quantiles are of interest => Later estimate VaRs for these models
  which.taus = c()
  for (k in 1:NES) {
    if (ES[k] < 0.50) {
      tau = 1 - ES[k]
      which.taus = c(which.taus, 1 - as.numeric(format(round(tau + (1:p-1)*(1-tau)/p,4),digits=4)))
    } else {
      # Which quantiles approximate ES?
      tau = ES[k]
      which.taus = c(which.taus, as.numeric(format(round(tau + (1:p-1)*(1-tau)/p,4),digits=4)))
    }
  }
  which.taus = unique(which.taus)
  NQ = length(which.taus)
  # Important to order from lowest to highest - for checking quantile crossing
  which.taus = which.taus[order(which.taus)]
  # From each estimation we can extract all quantiles, i.e. no need to re-estimate models for each quantile separately.
  
  
  
  
  
  
  # Number of GARCH model specifications - these are used to name models
  distrib.mod = c()
  # Initialize the lists
  specs <- list()
  converted_var_sets <- list()
  counter <- 1
  
  # Get the names of the sets
  set_names <- names(independant_var_sets)
  
  if (run_all_variations){
    for (i in seq_along(independant_var_sets)) {
      current_vars <- independant_var_sets[[i]]
      
      # Generate the specs
      specs[[paste0("EGARCH.", counter)]] <- (paste(depentant_var, "~", current_vars))
      
      # Convert the string set to a vector of variable names
      converted_vars <- convert_to_vector(current_vars)
      converted_var_sets[[paste0("set", i)]] <- converted_vars
      
      
      # Generate the model name
      model_name <- paste0("EGARCH_", independant_var_names[[i]])
      distrib.mod <- c(distrib.mod, model_name)
      counter <- counter + 1
    }
    
    # Generate the specs for each combination of sets
    for (n in 2:length(independant_var_sets)) {
      combinations <- combn(set_names, n, simplify = FALSE)
      
      for (combo in combinations) {
        current_vars <- paste(unlist(independant_var_sets[combo]), collapse = " + ")
        specs[[paste0("EGARCH.", counter)]] <- (paste(depentant_var, "~", current_vars))
        
        # Convert the string set to a vector of variable names
        converted_vars <- convert_to_vector(current_vars)
        converted_var_sets[[paste0("set", length(converted_var_sets)+1)]] <- converted_vars
        
        # Generate the model name
        model_name <- paste0("EGARCH_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        distrib.mod <- c(distrib.mod, model_name)
        counter <- counter + 1
      }
    }
    
    # Assign the generated names to the specs
    names(specs) <- distrib.mod
    
    
    
  }
  
  else if(run_single){
    current_vars <- single_var
    
    # Generate the specs
    specs[[paste0("EGARCH.", counter)]] <- (paste(depentant_var, "~", current_vars))
    
    # Convert the string set to a vector of variable names
    #converted_vars <- convert_to_vector(current_vars)
    #converted_var_sets[[paste0("set", i)]] <- converted_vars
    
    
    # Generate the model name
    model_name <- paste0("EGARCH_", single_var)
    distrib.mod <- c(distrib.mod, model_name)
    counter <- counter + 1
    names(specs) <- distrib.mod
    
  }
  
  
  else{
    # Generate the specs for each set on its own and convert variable sets
    for (i in seq_along(independant_var_sets)) {
      current_vars <- independant_var_sets[[i]]
      
      # Generate the specs
      specs[[paste0("EGARCH.", counter)]] <- (paste(depentant_var, "~", current_vars))
      
      # Convert the string set to a vector of variable names
      converted_vars <- convert_to_vector(current_vars)
      converted_var_sets[[paste0("set", i)]] <- converted_vars
      
      
      # Generate the model name
      model_name <- paste0("EGARCH_", independant_var_names[[i]])
      distrib.mod <- c(distrib.mod, model_name)
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
        specs[[paste0("EGARCH.", counter)]] <- (paste(depentant_var, "~", current_vars))
        
        # Convert the string set to a vector of variable names
        converted_vars <- convert_to_vector(current_vars)
        converted_var_sets[[paste0("set", length(converted_var_sets)+1)]] <- converted_vars
        
        # Generate the model name
        model_name <- paste0("EGARCH_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        distrib.mod <- c(distrib.mod, model_name)
        counter <- counter + 1
        # if (counter >= 2){
        #   break
        # }
      }
    }
    
    # Assign the generated names to the specs
    names(specs) <- distrib.mod
    
    
    
  }
  
  
  
  
  ND = length(distrib.mod);
  
  
  # Loop over garch specifications
  for (d in 1:ND) {
    print(distrib.mod[d])
    # Results frame: Observations x Number of Quantiles
    #Må dobbeltskjekke om det er riktig at denne koden skal kjøres.
    mat.taus = matrix(NA,nrow=TT,ncol=NQ)
    colnames(mat.taus) = paste(distrib.mod[d],'_',distrib,'_',round(which.taus,4),sep='')
    res = data.frame(Date=dt$Date,Realized=dt[,'Change'],mat.taus)
    # # Define specification

    
 
    # External regressor
    curr_set_name <- paste0("set", d)
    xre = as.xts(dt[, converted_var_sets[[curr_set_name]]], order.by = dt$Date)
    spec = ugarchspec(variance.model=list(model='eGARCH',garchOrder=c(1,1),
                                          external.regressors=xre),
                      mean.model=list(armaOrder=c(1,0),include.mean=TRUE),
                      distribution.model = distrib)
    

    # Dependent variable to model - returns
    ret = as.xts(dt$Change,order.by=dt$Date)
    
    # Rolling forecasts
    cl = makeCluster(nc)
    registerDoParallel(cl)
    TMP = foreach (i = EW:(TT-1), .packages = 'rugarch') %dopar% {
      # Estimate model
      
      fit = tryCatch(
        {
          ugarchfit(spec = spec, out.sample = 1, data= ret[(i-EW+1):(i+1)],
                    solver = 'hybrid') 
        },
        error=function(cond) {
          return(NA)
        }
      )
      if (is.na(fit)) return(rep(NA,length(which.taus)))
      return(fit)
      
    }
    stopCluster(cl)
    
    TES = matrix(NA,nrow=dim(dt)[1],ncol=NES)
    # Handling VaR, approx ES and full ES & ERRORS
    for (i in 1:length(TMP)) {
      if (any(is.na(TMP[[i]]))) {
        res[EW+i,-c(1:2)] = res[EW+i-1,-c(1:2)]
        TES[EW+i,]        = TES[EW+i-1,]
      } else {
        
        # Make forecast
        fit = TMP[[i]]
        ufor = ugarchforecast(fitORspec=fit,n.ahead=1)
        # Extract quantiles
        x = c(); for (r in 1:NQ) x[r] = as.numeric(quantile(ufor,probs=which.taus[r]))
        # Sanity filter
        if (x[1] < -5.5 | x[length(which.taus)] > 5.5) {
          res[EW+i,-c(1:2)] = res[EW+i-1,-c(1:2)]
          TES[EW+i,]        = TES[EW+i-1,]
          next
        }
        res[EW+i,-c(1:2)] = x
        
        # Exact Expected Shortfalls - not approximated via quantiles
        # We loop over ES
        for (k in 1:NES) {
          if (ES[k] < 0.5) {
            f = function(x,distrib,fit) qdist(distrib, p=x, mu = 0, sigma = 1, 
                                              skew  = coef(fit)["skew"], shape=coef(fit)["shape"])
            TES[EW+i,k] = fitted(ufor) + sigma(ufor) * integrate(f, 0, ES[k],distrib,fit)$value/ES[k]
          } else {
            f = function(x,distrib,fit) qdist(distrib, p=x, mu = 0, sigma = 1, 
                                              skew  = coef(fit)["skew"], shape=coef(fit)["shape"])
            TES[EW+i,k] = fitted(ufor) + sigma(ufor) * integrate(f, ES[k], 1, distrib, fit)$value/(1-ES[k])
          }
        }
        rm(x,fit,ufor) 
      }
    }
    
    # Sanity filter 
    for (i in (EW+2):TT) {
      # Test lower bound
      cond = which(res[i,-c(1:2)] < res[i-1,-c(1:2)] - 6*sd(dt$Change[(i-EW+1):i],na.rm=T))
      if (length(cond)>0) {
        res[i,-c(1:2)] = res[i-1,-c(1:2)]
      }
      # Test upper bound
      cond = which(res[i,-c(1:2)] > res[i-1,-c(1:2)] + 6*sd(dt$Change[(i-EW+1):i],na.rm=T))
      if (length(cond)>0) {
        res[i,-c(1:2)] = res[i-1,-c(1:2)]
      }
      # Check positiveness/negativeness & Low or High Var should not be near Median
      for (z in 1:NQ) {
        if (which.taus[z] < 0.5) {
          # positiveness of low quantiles (not good)
          if (res[i,z+2] > 0) res[i,z+2] = 0 - 0.0001
          # improbable values
          if (res[i,z+2] < quantile(res[(i-EW+1):(i-1),2],p=0.65,na.rm=T) & res[i,z+2] > quantile(res[(i-EW+1):(i-1),2],p=0.35,na.rm=T)) res[i,z+2] = res[i-1,z+2]
        } else {
          # positiveness of low quantiles (not good)
          if (res[i,z+2] < 0) res[i,z+2] = 0 + 0.0001
          # improbable values
          if (res[i,z+2] < quantile(res[(i-EW+1):(i-1),2],p=0.65,na.rm=T) & res[i,z+2] > quantile(res[(i-EW+1):(i-1),2],p=0.35,na.rm=T)) res[i,z+2] = res[i-1,z+2]
        }
      }
    }
    
    # Check quantile crossing
    # Quantiles have to be ordered from lowest to highest
    for (i in (EW+2):TT) for (z in 2:NQ) if (res[i,z+2] - res[i,z+1] < 0) res[i,z+2] = res[i,z+1] + 0.0001
    
    # We loop over ES - approximate ES
    for (k in 1:NES) {
      # Having all VaRs already - we need to find ES for the correct quantiles
      # Which quantiles are related to that ES?
      if (ES[k] < 0.50) {
        tau = 1 - ES[k]
        which.t = 1 - as.numeric(format(round(tau + (1:p-1)*(1-tau)/p,4),digits=4))
      } else {
        # Which quantiles approximate ES?
        tau = ES[k]
        which.t = as.numeric(format(round(tau + (1:p-1)*(1-tau)/p,4),digits=4))
      }
      which.t = which.t[order(which.t)]
      
      # Where is the name of those quantiles?
      nms = paste(distrib.mod[d],'_',distrib,'_',round(which.t,4),sep='')
      # Where are those variables?
      idx = which(names(res) %in% nms)
      
      # Calculate raw ES
      res$ES = apply(res[,idx],1,mean,na.rm=T)
      names(res)[which(names(res) == 'ES')] = paste(distrib.mod[d],'_',distrib,'_AES_',ES[k],sep='')
    }
    
    nms = c(); for (k in 1:NES) nms[k] = paste(distrib.mod[d],'_',distrib,'_ES_',ES[k],sep='')
    colnames(TES) = nms
    
    dt = data.frame(dt,res[,-c(1:2)],TES)
  }
  
  results = list()
  results[['data']] = dt
  return(results)

}