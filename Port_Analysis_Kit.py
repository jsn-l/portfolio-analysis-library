import pandas as pd
import numpy as np
from scipy.optimize import minimize

def drawdown(returns: pd.Series):
    
    """ 
        Takes a time series of returns, calculates dataframe with wealth, previous peaks and drawdown in %
    """
    
    w_index = 1000*(returns+1).cumprod()
    prev_peaks = w_index.cummax()
    drawdowns = (w_index - prev_peaks)/prev_peaks
    
    return pd.DataFrame({
        "Wealth" : w_index,
        "Peaks" : prev_peaks,
        "Drawdown" : drawdowns
    })


def semi_deviation(returns):
    """
    Returns the semideviation aka negative semideviation of r
    """

    is_neg = returns < 0
    return returns[is_neg].std(ddof=0)


def skew_kurt(returns,stat='skew'):
    """
     Computes the skewness / kurtosis of the dataframe
    """

    demeaned_ret = returns - returns.mean()
    sigma_ret = returns.std(ddof=0)
    
    if stat = 'skew':
        exp = (demeaned_ret**3).mean()
        result = exp/sigma_ret**3
    else:
        exp = (demeaned_ret**4).mean()
        result = exp/sigma_ret**4
    return result
   


def isnormal(ret,level=0.01):
    
    """
    Applies the JB test to determine if a series is normal or not
    Test is applied at the 1% level by default
    True if hypothesis of normality is accepted, false otherwise
    """
    import scipy.stats

    statistic, p_value = scipy.stats.jarque_bera(ret)
    return p_value > level
   

def var_historic(ret,level=5):
    """
        Historical VaR
    """
    import numpy as np
    
    if isinstance(ret,pd.DataFrame):
        return ret.aggregate(var_historic, level=level)
    elif isinstance(ret,pd.Series):
        return -np.percentile(ret,level)
    else:
        raise TypeError("Expected ret to be Series or DataFrame")
    
    
def var_gaussian(ret,level=5, Modified=False):
    from scipy.stats import norm
    """
        Parametric Gaussian VaR of a Series of Dataframe
    """
    z = norm.ppf(level/100)
    
    if Modified:
        s = skewness(ret)
        k = kurtosis(ret)
        z = (z+
             (z**2-1)*s/6 +
             (z**3-3*z)*(k-3)/24 -
             (2*z**3-5*z)*(s**2)/36
            )
    return -(ret.mean() + z*ret.std(ddof=0))
    
def cvar_historic(ret,level=5):
    """
    Computes the conditional VaR of a Series or a DataFrame
    """
    if isinstance(ret,pd.Series):
        
        # Find all returns less than the historical var and gives you a mask
        is_beyond = ret <= -var_historic(ret,level = level)
        
        # return the list with the mask applied and find the mean of this
        return -ret[is_beyond].mean()
    elif isinstance(ret,pd.DataFrame):
        return ret.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected ret to be a Series or Dataframe")

def annualized_rets(returns, periods_per_year):
    """
        Annualize a set of returns given periods per year
    """
    compounded_growth = (1+returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualized_vol(returns,periods_per_year):
    """
      Annualized the vol of a set of returns given periods per year
    """
    return returns.std()*(periods_per_year**0.5)

def sharpe_ratio(returns, riskfree_rate, periods_per_year):
    """
        Computes the annualised sharpe ratio of a set of returns
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = returns - rf_per_period
    ann_ex_ret = annualized_rets(excess_ret,periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    return ann_ex_ret/ann_vol
        
def portfolio_return(weights,returns):
        """
            Weights -> Returns
        """
        return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
        Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5        
        
def plot_ef2(n_points,er,cov,style=".-"):
    """
        Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })
    return ef.plot.line(x="Volatility",y="Returns",style=style)

def minimize_vol(target_return,er,cov):
    """
        Target return -> Weight Vector
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target = {
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_vol,init_guess,
                       args=(cov,),method="SLSQP",
                       options={'disp':False},
                       constraints=(return_is_target,weights_sum_to_1),
                       bounds=bounds
                        )
    return results.x

def optimal_weights(n_points,er,cov):
    """
        Generates list of weights to run optimizer on min vol
    """
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights
    
def msr(riskfree_rate,er,cov):
    """
        Risk free rate + ER + COV -> Weight Vector
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    
    def neg_sharpe_ratio(weights,riskfree_rate, er, cov):
        """
            Returns the negative of the sharpe
        """
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio,init_guess,
                       args=(riskfree_rate,er,cov,),method="SLSQP",
                       options={'disp':False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                        )
    return results.x    

   
def gmv(cov):
    """
        Returns weights of the global minimum vol portfolio
        given the covariance matrix.
    """
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
    
def plot_ef(n_points,er,cov,show_cml=False,style=".-",riskfree_rate=0,show_ew=False,show_gmv=False):
    """ 
        Plots the multi-asset efficient frontier
    """ 
    weights = optimal_weights(n_points,er,cov)
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })
    ax = ef.plot.line(x="Volatility",y="Returns",style=style)
    
    if show_ew:
        n=er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        vol_ew = portfolio_vol(w_ew,cov)     
        

        ax.plot(vol_ew,r_ew,color="goldenrod",marker="o",markersize=10)
             
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
     
        ax.plot(vol_gmv,r_gmv,color="midnightblue",marker="o",markersize=10)
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate,er,cov)
        r_msr = portfolio_return(w_msr,er)
        vol_msr = portfolio_vol(w_msr,cov)

        cml_x = [0,vol_msr]
        cml_y = [riskfree_rate ,r_msr]
        
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)

    return ax
        

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualized_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualized_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, Modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
        