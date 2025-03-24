#### File implementing Diagnostc tests on the data ####


# Should implement the following tests:
# 1. Augmented Dikcy Fuller test - for stationarity (test on each stock)
# 2. Ljung-Box or Breusch-Godfrey for serial correlation (test on each stock)
# 3. ARCH-LM (Engle) test for conditional heteroskedasticity (test on each stock)
# 4. Durbin-Watson statistic for residual autocorrelation (test on each stock)
# 5. Jarque-Bera test for normality (test on each stock)
# 6 VIF (Variance Inflation Factor) for multicollinearity (test on each stock)


# %%
# import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import reset_ramsey