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




# quantile models - with IV
QuantileVaR = function(dt = D, EW = 1500, p = 5, ES = ES, nc = 24, dep_var = depentant_var, run_all_variations = FALSE) {
  library(xts)
  library(foreach)
  library(doParallel)
  library(quantreg)
  
  # Number of observations
  TT = dim(dt)[1]
  # Number of out-of-sample observations
  TF = TT - EW
  # Number of ES quantiles
  NES = length(ES)
  
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
      specs[[paste0("QHAR.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
      
      # Generate the model name
      model_name <- paste0("QHAR_", independant_var_names[[i]])
      model_names <- c(model_names, model_name)
      counter <- counter + 1
    }
    
    # Generate the specs for each combination of sets
    for (n in 2:length(independant_var_sets)) {
      combinations <- combn(set_names, n, simplify = FALSE)
      
      for (combo in combinations) {
        current_vars <- paste(unlist(independant_var_sets[combo]), collapse = " + ")
        specs[[paste0("QHAR.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
        
        # Generate the model name
        model_name <- paste0("QHAR_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        
        
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
      specs[[paste0("QHAR.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
      
      # Generate the model name
      model_name <- paste0("QHAR_", independant_var_names[[i]])
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
        specs[[paste0("QHAR.", counter)]] <- as.formula(paste(depentant_var, "~", current_vars))
        
        # Generate the model name
        model_name <- paste0("QHAR_", paste(unlist(independant_var_names_set[combo]), collapse = "_"))
        model_names <- c(model_names, model_name)
        counter <- counter + 1
      }
    }
    
    # Assign the generated names to the specs
    names(specs) <- model_names
    
    
  }
  
  # Number of specifications
  ND = length(specs);
  
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
  
  # Loop over specifications
  for (d in 1:ND) {
    print(specs[d])
    
    # Results frame: Observations x Number of Quantiles
    mat.taus = matrix(NA,nrow=TT,ncol=NQ)
    
    colnames(mat.taus) = paste(names(specs)[d],'_',round(which.taus,4),sep='')
    res = data.frame(Date=dt$Date,Realized=dt[,'Change'],mat.taus)
    
    # Select specification
    spec = specs[[d]]
    
    ret = as.xts(dt$Change,order.by=dt$Date)
    
    # Rolling VaR forecasts - standard quantile regression
    cl = makeCluster(nc)
    registerDoParallel(cl)
    TMP = foreach (i = EW:(TT-1),.packages = 'quantreg') %dopar% {
      print(i)
      # Now estimate the model for all quantiles
      A = coefficients(rq(spec,tau=which.taus,data=dt[(i-EW+1):(i),all.vars(spec)]))
      # prediction
      return(c(1,as.numeric(dt[(i+1),all.vars(spec)[-1]])) %*% A)
      # as.numeric(c(1,as.numeric(dt[(i+1),all.vars(spec)[-1]])) %*% as.matrix(A$bhat))
    }
    stopCluster(cl)
    
    # VaR extraction
    for (i in 1:length(TMP)) res[(EW+i),-c(1:2)] = TMP[[i]]
    
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
    
    # Having quantiles - we loop over different Expected Shortfalls
    # We loop over ES
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
      nms = paste(names(specs)[d],'_',round(which.t,4),sep='')
      # Where are those variables?
      idx = which(names(res) %in% nms)
      
      # Calculate raw ES
      res$ES = apply(res[,idx],1,mean,na.rm=T)
      names(res)[which(names(res) == 'ES')] = paste(names(specs)[d],'_ES_',ES[k],sep='')
      
    }
    dt = data.frame(dt,res[,-c(1:2)])
  }
  
  results = list()
  results[['data']] = dt
  return(results)
}

