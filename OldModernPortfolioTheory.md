
# Old Modern portfolio theory

## Efficient frontier, VaR, Expected Shortfall, Bootstrap, Monte-Carlo

In this tutorial, we're going to calculate the efficient frontier based on historical and forecasted data, and then generate some forward-looking returns.

As a starting point we'll use returns of 12 asset classes, namely developed markets bonds(FI.DEV), developed markets equities(EQ.DEV), emerging market bonds (FI.EM), corporate bonds(FI.CORP), emerging market equities(EQ.EM), high yield bonds(FI.HY), inflation-linked bonds(FI.IL), hedge funds(HF), real estate securities(RE.SEC), commodities(COMMOD), private equity(PRIV.EQ), bills(CASH).


```python
#loadind required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import cvxopt as opt
from cvxopt import blas, solvers
```


```python
#some formatting
pd.options.display.float_format = '{:.02%}'.format #this is to format pandas dataframes nicely
#pd.options.display.float_format = '{:.4f}'.format #this is to format pandas dataframes nicely
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" #this is just to show all output for any cell, not the last operator output
solvers.options['show_progress'] = False # Turn off progress printing
```


```python
myPath = r'D:\Serega\Education\!Interviews\Portfolio\SAA_portfolio\Data_Source.xlsx'
```


```python
returns = pd.read_excel(myPath, index_col=0)
```


```python
returns.head(2)
print('...')
returns.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1998-02-28</th>
      <td>0.66%</td>
      <td>6.77%</td>
      <td>-0.14%</td>
      <td>0.37%</td>
      <td>10.44%</td>
      <td>0.45%</td>
      <td>1.26%</td>
      <td>-3.17%</td>
      <td>-5.49%</td>
      <td>-5.26%</td>
      <td>8.02%</td>
      <td>0.37%</td>
    </tr>
    <tr>
      <th>1998-03-31</th>
      <td>-0.88%</td>
      <td>4.33%</td>
      <td>0.90%</td>
      <td>-0.50%</td>
      <td>4.23%</td>
      <td>0.96%</td>
      <td>2.31%</td>
      <td>-2.91%</td>
      <td>-0.75%</td>
      <td>0.98%</td>
      <td>2.80%</td>
      <td>0.49%</td>
    </tr>
  </tbody>
</table>
</div>



    ...
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-10-31</th>
      <td>-0.16%</td>
      <td>7.96%</td>
      <td>2.69%</td>
      <td>0.61%</td>
      <td>7.14%</td>
      <td>2.99%</td>
      <td>0.69%</td>
      <td>-1.68%</td>
      <td>-6.25%</td>
      <td>1.14%</td>
      <td>7.08%</td>
      <td>-0.03%</td>
    </tr>
    <tr>
      <th>2015-11-30</th>
      <td>-1.72%</td>
      <td>-0.44%</td>
      <td>-0.82%</td>
      <td>-1.10%</td>
      <td>-3.90%</td>
      <td>-2.04%</td>
      <td>-1.24%</td>
      <td>-0.44%</td>
      <td>-2.00%</td>
      <td>-7.51%</td>
      <td>4.82%</td>
      <td>-0.01%</td>
    </tr>
  </tbody>
</table>
</div>



As per the output above, in the input file we have monthly returns for a number of assets from February 1998 to November 2015. It is a good data range because it includes the dotcom crysis, the mortgage buble and consequent recoveries. You can choose your own time horizon. 
If you want do download other data from the internet there is a number of packages to do that. Just don't forget to convert price data to returns.
Let's plot this returns to see relative performance of assets. 


```python
def cumulative_returns(returns):
    cum_ret = returns + 1
    for i in range(1,returns.shape[0]):
        cum_ret.iloc[i,:] = cum_ret.iloc[i,:]*cum_ret.iloc[i-1,:]
    return cum_ret - 1

def plot_returns(cum_ret, legend = True):
    plt.figure()
    if (legend):
        cum_ret.plot(figsize=(12, 6))
    else:
        cum_ret.plot(figsize=(12, 6), legend=None)
    plt.title('Cumulative return of assets')
    if (legend):
        plt.legend(loc='upper left')
    #else:
    #    ax.get_legend().remove()
        #plt.legend(loc='upper right')   
    plt.xlabel('Time')
    plt.ylabel('Cumilative return, %')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.grid(True)
```


```python
cum_ret = cumulative_returns(returns)
```


```python
InteractiveShell.ast_node_interactivity = "last"

plot_returns(cum_ret)
```


    <Figure size 432x288 with 0 Axes>



![png](output_10_1.png)


The worst performing classes are hedge funds and real estate securities. Maybe the indices chosen are not representative. However, since it is only an exercise, we'll leave averything as it is.
Let's calculate parameters of these returns.


```python
#function for historical VaR and CVaR calculation
def __return_sorted_columns(df):
    sorted_df = pd.DataFrame(columns=df.columns)
    for col in df:
        sorted_df[col] = sorted(df[col])
    return sorted_df

def var_historical(rtns, confidence=.95):
    sorted_rtns = __return_sorted_columns(rtns)
    ind = int(np.floor(len(rtns)*(1-confidence))) #better to take lower value to overestimate the risk than to underestimate it
    return sorted_rtns.iloc[ind-1]

def cvar_historical(rtns, confidence=.95):
    sorted_rtns = __return_sorted_columns(rtns)
    ind = int(np.floor(len(rtns)*(1-confidence))) #better to take lower value to overestimate the risk than to underestimate it
    return np.mean(sorted_rtns[0:ind])

def var_analytical(rtns, confidence=.95):
    mu = rtns.mean() # in some cases mean return may assumed to be zero
    std = rtns.std()
    return mu - std*norm.ppf(confidence)

def cvar_analytical(rtns, confidence=.95):
    mu = rtns.mean() # in some cases mean return may assumed to be zero
    std = rtns.std()
    return mu - std*norm.pdf(norm.ppf(confidence))/(1-confidence)


def calculateparameters(rtns, confidence=.95):
    """This function returns Mean return, Standard deviation, Historical VaR, Historical CVaR, Analytical VaR, Analytical CVaR
    Parameters
    ----------
    rtns (pandas dataframe): asset returns
    """
    mean_asset_rtn = rtns.mean()
    std_asset_rtn = rtns.std()
    VaR_hist = var_historical(rtns, confidence)
    CVaR_hist = cvar_historical(rtns, confidence)
    VaR_covar = var_analytical(rtns, confidence)
    CVaR_covar = cvar_analytical(rtns, confidence)
    params = pd.concat([mean_asset_rtn, std_asset_rtn,VaR_hist, CVaR_hist, VaR_covar, CVaR_covar], axis=1)
    params = params.transpose()
    params.index = ['Mean return','Standard deviation', 'Historical VaR', 'Historical CVaR', 
                    'Analytical VaR', 'Analytical CVaR']
    return params
```


```python
calculateparameters(returns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean return</th>
      <td>0.38%</td>
      <td>0.57%</td>
      <td>0.52%</td>
      <td>0.46%</td>
      <td>0.82%</td>
      <td>0.56%</td>
      <td>0.52%</td>
      <td>-0.52%</td>
      <td>-0.23%</td>
      <td>0.53%</td>
      <td>0.87%</td>
      <td>0.21%</td>
    </tr>
    <tr>
      <th>Standard deviation</th>
      <td>1.82%</td>
      <td>4.58%</td>
      <td>1.88%</td>
      <td>1.82%</td>
      <td>6.98%</td>
      <td>2.91%</td>
      <td>2.16%</td>
      <td>2.02%</td>
      <td>5.14%</td>
      <td>6.70%</td>
      <td>7.38%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <th>Historical VaR</th>
      <td>-2.97%</td>
      <td>-8.53%</td>
      <td>-2.82%</td>
      <td>-2.54%</td>
      <td>-10.48%</td>
      <td>-3.97%</td>
      <td>-3.40%</td>
      <td>-3.48%</td>
      <td>-6.99%</td>
      <td>-11.23%</td>
      <td>-11.30%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>Historical CVaR</th>
      <td>-3.59%</td>
      <td>-10.94%</td>
      <td>-4.65%</td>
      <td>-4.04%</td>
      <td>-16.28%</td>
      <td>-7.21%</td>
      <td>-4.95%</td>
      <td>-4.47%</td>
      <td>-8.96%</td>
      <td>-14.55%</td>
      <td>-16.85%</td>
      <td>-0.01%</td>
    </tr>
    <tr>
      <th>Analytical VaR</th>
      <td>-2.61%</td>
      <td>-6.96%</td>
      <td>-2.58%</td>
      <td>-2.53%</td>
      <td>-10.66%</td>
      <td>-4.22%</td>
      <td>-3.02%</td>
      <td>-3.84%</td>
      <td>-8.69%</td>
      <td>-10.49%</td>
      <td>-11.28%</td>
      <td>-0.12%</td>
    </tr>
    <tr>
      <th>Analytical CVaR</th>
      <td>-3.37%</td>
      <td>-8.88%</td>
      <td>-3.36%</td>
      <td>-3.29%</td>
      <td>-13.58%</td>
      <td>-5.44%</td>
      <td>-3.93%</td>
      <td>-4.69%</td>
      <td>-10.83%</td>
      <td>-13.29%</td>
      <td>-14.36%</td>
      <td>-0.21%</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, historical VaR slightly overestimates the risk. It happens because we round the index of the historical return correspondent to the chosen confidence level.


We can generate expected returns using bootstrap or covariance based Monte-Carlo.


```python
def montecarlo(rtns, num_simulations = 10000, seed=1):
    '''Covariance based Monte-Carlo, returns are assumed to be normally distributed
    '''
    np.random.seed(seed)
    
    n_assets = rtns.shape[1]
    mean_asset_rtn = rtns.mean()
    std_asset_rtn = rtns.std()
    rand_rtns = (np.random.normal(size=num_simulations*n_assets)).reshape(num_simulations,n_assets)
    if (n_assets!=1):
        cormat = rtns.corr()
        cholesky_decomposition = (np.linalg.cholesky(cormat)).transpose()
        zscore = np.dot(rand_rtns, cholesky_decomposition)
        rtns_simulations = pd.DataFrame(columns=rtns.columns)
    else:
        zscore = rand_rtns
        rtns_simulations = pd.DataFrame(columns=['The_port'])
    
    #haven't found an elegant way to do this. Ended up with a loop. There should be some convenient function in numpy or pandas...
    for i in range(zscore.shape[0]):
        rtns_simulations.loc[i] = mean_asset_rtn + np.multiply(zscore[i,:],std_asset_rtn)
    return rtns_simulations

def bootstrap(rtns, num_simulations = 10000, chunksize = 3, seed=1):
    '''Takes historical data to generate returns
    '''
    n_returns = rtns.shape[0]
    if (chunksize<1):
        chunksize = 1
        print('Chunksize cannot be negative. chunksize is assumed to be 1')
        
    returns_local = rtns.append(rtns.iloc[0:(chunksize-1),:]) #this is to be able to take pieces from the end of the series
    chunks = num_simulations//chunksize
    rtns_simulations = pd.DataFrame(columns=rtns.columns)
    np.random.seed(seed)
    for idx in np.random.choice(n_returns, size=chunks, replace=True):
        rtns_simulations = rtns_simulations.append(returns_local.iloc[idx:(idx+chunksize),:])
    
    #adding variables which are lower than 
    fraction_period = num_simulations%chunksize
    if fraction_period:
        idx = np.random.randint(n_returns)
        rtns_simulations = rtns_simulations.append(returns_local.iloc[idx:(idx+fraction_period),:])
    
    rtns_simulations.index = range(num_simulations)
    
    return rtns_simulations
```


```python
bootstrap_returns = bootstrap(returns)
montecarlo_returns = montecarlo(returns)
```

Parameters of returns generated with bootstrap:


```python
calculateparameters(bootstrap_returns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean return</th>
      <td>0.38%</td>
      <td>0.61%</td>
      <td>0.51%</td>
      <td>0.45%</td>
      <td>0.82%</td>
      <td>0.53%</td>
      <td>0.52%</td>
      <td>-0.53%</td>
      <td>-0.27%</td>
      <td>0.51%</td>
      <td>0.78%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <th>Standard deviation</th>
      <td>1.82%</td>
      <td>4.55%</td>
      <td>1.89%</td>
      <td>1.81%</td>
      <td>6.88%</td>
      <td>2.91%</td>
      <td>2.16%</td>
      <td>2.01%</td>
      <td>5.09%</td>
      <td>6.69%</td>
      <td>7.36%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <th>Historical VaR</th>
      <td>-2.83%</td>
      <td>-8.42%</td>
      <td>-2.75%</td>
      <td>-2.52%</td>
      <td>-9.99%</td>
      <td>-3.94%</td>
      <td>-3.03%</td>
      <td>-3.45%</td>
      <td>-6.84%</td>
      <td>-10.65%</td>
      <td>-11.07%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>Historical CVaR</th>
      <td>-3.53%</td>
      <td>-10.67%</td>
      <td>-4.52%</td>
      <td>-3.92%</td>
      <td>-15.51%</td>
      <td>-7.13%</td>
      <td>-4.86%</td>
      <td>-4.48%</td>
      <td>-8.56%</td>
      <td>-14.45%</td>
      <td>-16.81%</td>
      <td>-0.01%</td>
    </tr>
    <tr>
      <th>Analytical VaR</th>
      <td>-2.61%</td>
      <td>-6.88%</td>
      <td>-2.60%</td>
      <td>-2.53%</td>
      <td>-10.49%</td>
      <td>-4.26%</td>
      <td>-3.03%</td>
      <td>-3.84%</td>
      <td>-8.65%</td>
      <td>-10.50%</td>
      <td>-11.33%</td>
      <td>-0.12%</td>
    </tr>
    <tr>
      <th>Analytical CVaR</th>
      <td>-3.37%</td>
      <td>-8.79%</td>
      <td>-3.39%</td>
      <td>-3.28%</td>
      <td>-13.36%</td>
      <td>-5.47%</td>
      <td>-3.94%</td>
      <td>-4.68%</td>
      <td>-10.78%</td>
      <td>-13.30%</td>
      <td>-14.40%</td>
      <td>-0.21%</td>
    </tr>
  </tbody>
</table>
</div>



Parameters of returns generated with Monte-Carlo:


```python
calculateparameters(montecarlo_returns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean return</th>
      <td>0.37%</td>
      <td>0.49%</td>
      <td>0.52%</td>
      <td>0.46%</td>
      <td>0.80%</td>
      <td>0.56%</td>
      <td>0.52%</td>
      <td>-0.51%</td>
      <td>-0.16%</td>
      <td>0.57%</td>
      <td>0.78%</td>
      <td>0.21%</td>
    </tr>
    <tr>
      <th>Standard deviation</th>
      <td>1.83%</td>
      <td>4.56%</td>
      <td>1.89%</td>
      <td>1.82%</td>
      <td>6.93%</td>
      <td>2.91%</td>
      <td>2.16%</td>
      <td>2.01%</td>
      <td>5.15%</td>
      <td>6.69%</td>
      <td>7.43%</td>
      <td>0.20%</td>
    </tr>
    <tr>
      <th>Historical VaR</th>
      <td>-2.63%</td>
      <td>-6.85%</td>
      <td>-2.59%</td>
      <td>-2.54%</td>
      <td>-10.73%</td>
      <td>-4.20%</td>
      <td>-3.02%</td>
      <td>-3.79%</td>
      <td>-8.62%</td>
      <td>-10.54%</td>
      <td>-11.32%</td>
      <td>-0.12%</td>
    </tr>
    <tr>
      <th>Historical CVaR</th>
      <td>-3.44%</td>
      <td>-8.76%</td>
      <td>-3.41%</td>
      <td>-3.24%</td>
      <td>-13.56%</td>
      <td>-5.42%</td>
      <td>-3.90%</td>
      <td>-4.64%</td>
      <td>-10.74%</td>
      <td>-13.15%</td>
      <td>-14.39%</td>
      <td>-0.20%</td>
    </tr>
    <tr>
      <th>Analytical VaR</th>
      <td>-2.63%</td>
      <td>-7.00%</td>
      <td>-2.59%</td>
      <td>-2.52%</td>
      <td>-10.59%</td>
      <td>-4.23%</td>
      <td>-3.03%</td>
      <td>-3.81%</td>
      <td>-8.62%</td>
      <td>-10.44%</td>
      <td>-11.43%</td>
      <td>-0.12%</td>
    </tr>
    <tr>
      <th>Analytical CVaR</th>
      <td>-3.39%</td>
      <td>-8.91%</td>
      <td>-3.38%</td>
      <td>-3.28%</td>
      <td>-13.49%</td>
      <td>-5.44%</td>
      <td>-3.93%</td>
      <td>-4.65%</td>
      <td>-10.77%</td>
      <td>-13.24%</td>
      <td>-14.53%</td>
      <td>-0.20%</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, generated returns have almost the same parameters as our initial sample, which confirms that the generation functions work correctly.

Let's visualize the result, people love it. We can make a density plot for returns of equities.


```python
equity_returns_for_plotting = pd.concat([montecarlo_returns['EQ.DEV'], bootstrap_returns['EQ.DEV']],axis=1)
equity_returns_for_plotting.columns=['Monte-Carlo Equity Returns','Bootstrap  Equity Returns']
fig, ax = plt.subplots(figsize=(12, 8))
equity_returns_for_plotting.plot.kde(ax=ax, legend=True, title='Monte-Carlo and Bootstrap equity returns')
#equity_returns_for_plotting.plot.hist(density=True, ax=ax)
ax.grid(axis='x')
ax.grid(axis='y')
#ax.set_facecolor('#d8dcd6')
```


![png](output_23_0.png)


The return generated by Monte-Carlo is more smooth. Seems like bootstrap returns have fatter tails.

Let's annualize our monthly returns. We'll proceed with bootstrap returns.


```python
def returns_period_upscale(rtns, periodicity = 12, annualize_last = True):
    new_returns = pd.DataFrame(columns = rtns.columns)
    rtns += 1
    n_steps = rtns.shape[0]//periodicity
    
    for i in range(n_steps):
        new_returns.loc[i] = np.prod(rtns.iloc[(i*periodicity):((i+1)*periodicity)],axis=0)
    
    fraction_period = rtns.shape[0]%periodicity
    if fraction_period:
        new_returns.loc[n_steps] = np.prod(rtns.iloc[(n_steps*periodicity):],axis=0)
        if annualize_last: new_returns.loc[n_steps] = np.power(new_returns.loc[n_steps],periodicity/fraction_period)
    
    rtns -= 1 #python passes this dataframe by reference, and we don't want the internal function to make changes. I should've copied this Dataframe at the beginning of the function, and work with the copy. But I don't want to)
    
    return new_returns-1
```


```python
annual_returns = returns_period_upscale(bootstrap_returns)
```


```python
covmat, corrmat = [returns.cov(), returns.corr()]
corrmat.style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col1 {
            background-color:  #d8dce2;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col2 {
            background-color:  #f5c4ac;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col3 {
            background-color:  #e9785d;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col4 {
            background-color:  #dadce0;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col5 {
            background-color:  #d3dbe7;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col6 {
            background-color:  #ee8468;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col7 {
            background-color:  #d7dce3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col8 {
            background-color:  #dcdddd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col9 {
            background-color:  #bbd1f8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col10 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col11 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col0 {
            background-color:  #7da0f9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col2 {
            background-color:  #edd2c3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col3 {
            background-color:  #f2cbb7;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col4 {
            background-color:  #d24b40;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col5 {
            background-color:  #e9785d;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col6 {
            background-color:  #e3d9d3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col7 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col8 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col9 {
            background-color:  #ead5c9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col10 {
            background-color:  #e57058;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col11 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col0 {
            background-color:  #ead4c8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col1 {
            background-color:  #f7b89c;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col3 {
            background-color:  #ee8468;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col4 {
            background-color:  #f7ad90;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col5 {
            background-color:  #f49a7b;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col6 {
            background-color:  #f39577;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col7 {
            background-color:  #93b5fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col8 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col9 {
            background-color:  #c1d4f4;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col10 {
            background-color:  #edd1c2;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col11 {
            background-color:  #7093f3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col0 {
            background-color:  #ee8468;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col1 {
            background-color:  #f7ad90;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col2 {
            background-color:  #ed8366;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col4 {
            background-color:  #f7b093;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col5 {
            background-color:  #f39778;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col6 {
            background-color:  #d44e41;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col7 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col8 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col9 {
            background-color:  #ead5c9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col10 {
            background-color:  #e1dad6;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col11 {
            background-color:  #7295f4;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col0 {
            background-color:  #7b9ff9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col1 {
            background-color:  #d24b40;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col2 {
            background-color:  #f1ccb8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col3 {
            background-color:  #efcebd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col5 {
            background-color:  #e97a5f;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col6 {
            background-color:  #e1dad6;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col7 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col8 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col9 {
            background-color:  #f2cab5;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col10 {
            background-color:  #ec7f63;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col11 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col0 {
            background-color:  #8badfd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col1 {
            background-color:  #e57058;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col2 {
            background-color:  #f7ac8e;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col3 {
            background-color:  #f7ac8e;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col4 {
            background-color:  #e67259;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col6 {
            background-color:  #f4c5ad;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col7 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col8 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col9 {
            background-color:  #ead4c8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col10 {
            background-color:  #f39577;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col11 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col0 {
            background-color:  #f18f71;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col1 {
            background-color:  #f7bca1;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col2 {
            background-color:  #f18f71;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col3 {
            background-color:  #d24b40;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col4 {
            background-color:  #f7ba9f;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col5 {
            background-color:  #f7ac8e;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col7 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col8 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col9 {
            background-color:  #ecd3c5;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col10 {
            background-color:  #dadce0;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col11 {
            background-color:  #779af7;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col0 {
            background-color:  #7597f6;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col2 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col4 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col7 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col8 {
            background-color:  #f4987a;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col9 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col10 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col11 {
            background-color:  #84a7fc;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col0 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col1 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col2 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col3 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col4 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col5 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col6 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col7 {
            background-color:  #f29274;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col8 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col9 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col10 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col11 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col0 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col1 {
            background-color:  #f6bea4;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col2 {
            background-color:  #bcd2f7;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col3 {
            background-color:  #e4d9d2;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col4 {
            background-color:  #f7af91;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col5 {
            background-color:  #f4c6af;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col6 {
            background-color:  #e3d9d3;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col7 {
            background-color:  #85a8fc;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col8 {
            background-color:  #b2ccfb;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col9 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col10 {
            background-color:  #ead5c9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col11 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col1 {
            background-color:  #e36c55;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col2 {
            background-color:  #d5dbe5;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col3 {
            background-color:  #bed2f6;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col4 {
            background-color:  #e9785d;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col5 {
            background-color:  #f4987a;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col6 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col7 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col8 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col9 {
            background-color:  #d2dbe8;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col10 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col11 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col0 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col1 {
            background-color:  #b6cefa;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col2 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col3 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col4 {
            background-color:  #c0d4f5;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col5 {
            background-color:  #9abbff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col6 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col7 {
            background-color:  #dddcdc;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col8 {
            background-color:  #dfdbd9;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col9 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col10 {
            background-color:  #adc9fd;
            color:  #000000;
        }    #T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col11 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_278b9158_8673_11e9_a6f7_80c5f2b6775c" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >FI.DEV</th>        <th class="col_heading level0 col1" >EQ.DEV</th>        <th class="col_heading level0 col2" >FI.EM</th>        <th class="col_heading level0 col3" >FI.CORP</th>        <th class="col_heading level0 col4" >EQ.EM</th>        <th class="col_heading level0 col5" >FI.HY</th>        <th class="col_heading level0 col6" >FI.IL</th>        <th class="col_heading level0 col7" >HF</th>        <th class="col_heading level0 col8" >RE.SEC</th>        <th class="col_heading level0 col9" >COMMOD</th>        <th class="col_heading level0 col10" >Private EQ</th>        <th class="col_heading level0 col11" >CASH</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row0" class="row_heading level0 row0" >FI.DEV</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col0" class="data row0 col0" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col1" class="data row0 col1" >0.053</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col2" class="data row0 col2" >0.47</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col3" class="data row0 col3" >0.76</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col4" class="data row0 col4" >0.053</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col5" class="data row0 col5" >0.1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col6" class="data row0 col6" >0.73</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col7" class="data row0 col7" >0.03</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col8" class="data row0 col8" >0.12</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col9" class="data row0 col9" >0.11</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col10" class="data row0 col10" >-0.19</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow0_col11" class="data row0 col11" >0.16</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row1" class="row_heading level0 row1" >EQ.DEV</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col0" class="data row1 col0" >0.053</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col1" class="data row1 col1" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col2" class="data row1 col2" >0.39</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col3" class="data row1 col3" >0.45</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col4" class="data row1 col4" >0.84</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col5" class="data row1 col5" >0.71</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col6" class="data row1 col6" >0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col7" class="data row1 col7" >-0.83</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col8" class="data row1 col8" >-0.63</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col9" class="data row1 col9" >0.35</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col10" class="data row1 col10" >0.72</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow1_col11" class="data row1 col11" >-0.15</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row2" class="row_heading level0 row2" >FI.EM</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col0" class="data row2 col0" >0.47</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col1" class="data row2 col1" >0.39</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col2" class="data row2 col2" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col3" class="data row2 col3" >0.72</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col4" class="data row2 col4" >0.43</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col5" class="data row2 col5" >0.58</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col6" class="data row2 col6" >0.68</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col7" class="data row2 col7" >-0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col8" class="data row2 col8" >-0.41</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col9" class="data row2 col9" >0.14</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col10" class="data row2 col10" >0.26</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow2_col11" class="data row2 col11" >0.0016</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row3" class="row_heading level0 row3" >FI.CORP</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col0" class="data row3 col0" >0.76</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col1" class="data row3 col1" >0.45</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col2" class="data row3 col2" >0.72</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col3" class="data row3 col3" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col4" class="data row3 col4" >0.42</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col5" class="data row3 col5" >0.59</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col6" class="data row3 col6" >0.88</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col7" class="data row3 col7" >-0.38</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col8" class="data row3 col8" >-0.22</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col9" class="data row3 col9" >0.35</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col10" class="data row3 col10" >0.16</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow3_col11" class="data row3 col11" >0.0062</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row4" class="row_heading level0 row4" >EQ.EM</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col0" class="data row4 col0" >0.053</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col1" class="data row4 col1" >0.84</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col2" class="data row4 col2" >0.43</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col3" class="data row4 col3" >0.42</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col4" class="data row4 col4" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col5" class="data row4 col5" >0.7</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col6" class="data row4 col6" >0.36</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col7" class="data row4 col7" >-0.86</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col8" class="data row4 col8" >-0.61</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col9" class="data row4 col9" >0.43</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col10" class="data row4 col10" >0.67</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow4_col11" class="data row4 col11" >-0.12</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row5" class="row_heading level0 row5" >FI.HY</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col0" class="data row5 col0" >0.1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col1" class="data row5 col1" >0.71</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col2" class="data row5 col2" >0.58</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col3" class="data row5 col3" >0.59</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col4" class="data row5 col4" >0.7</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col5" class="data row5 col5" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col6" class="data row5 col6" >0.5</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col7" class="data row5 col7" >-0.68</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col8" class="data row5 col8" >-0.57</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col9" class="data row5 col9" >0.36</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col10" class="data row5 col10" >0.58</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow5_col11" class="data row5 col11" >-0.2</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row6" class="row_heading level0 row6" >FI.IL</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col0" class="data row6 col0" >0.73</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col1" class="data row6 col1" >0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col2" class="data row6 col2" >0.68</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col3" class="data row6 col3" >0.88</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col4" class="data row6 col4" >0.36</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col5" class="data row6 col5" >0.5</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col6" class="data row6 col6" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col7" class="data row6 col7" >-0.32</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col8" class="data row6 col8" >-0.17</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col9" class="data row6 col9" >0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col10" class="data row6 col10" >0.11</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow6_col11" class="data row6 col11" >0.027</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row7" class="row_heading level0 row7" >HF</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col0" class="data row7 col0" >0.03</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col1" class="data row7 col1" >-0.83</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col2" class="data row7 col2" >-0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col3" class="data row7 col3" >-0.38</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col4" class="data row7 col4" >-0.86</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col5" class="data row7 col5" >-0.68</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col6" class="data row7 col6" >-0.32</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col7" class="data row7 col7" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col8" class="data row7 col8" >0.57</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col9" class="data row7 col9" >-0.44</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col10" class="data row7 col10" >-0.74</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow7_col11" class="data row7 col11" >0.071</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row8" class="row_heading level0 row8" >RE.SEC</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col0" class="data row8 col0" >0.12</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col1" class="data row8 col1" >-0.63</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col2" class="data row8 col2" >-0.41</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col3" class="data row8 col3" >-0.22</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col4" class="data row8 col4" >-0.61</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col5" class="data row8 col5" >-0.57</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col6" class="data row8 col6" >-0.17</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col7" class="data row8 col7" >0.57</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col8" class="data row8 col8" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col9" class="data row8 col9" >-0.12</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col10" class="data row8 col10" >-0.74</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow8_col11" class="data row8 col11" >0.15</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row9" class="row_heading level0 row9" >COMMOD</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col0" class="data row9 col0" >0.11</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col1" class="data row9 col1" >0.35</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col2" class="data row9 col2" >0.14</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col3" class="data row9 col3" >0.35</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col4" class="data row9 col4" >0.43</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col5" class="data row9 col5" >0.36</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col6" class="data row9 col6" >0.37</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col7" class="data row9 col7" >-0.44</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col8" class="data row9 col8" >-0.12</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col9" class="data row9 col9" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col10" class="data row9 col10" >0.22</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow9_col11" class="data row9 col11" >-0.045</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row10" class="row_heading level0 row10" >Private EQ</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col0" class="data row10 col0" >-0.19</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col1" class="data row10 col1" >0.72</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col2" class="data row10 col2" >0.26</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col3" class="data row10 col3" >0.16</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col4" class="data row10 col4" >0.67</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col5" class="data row10 col5" >0.58</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col6" class="data row10 col6" >0.11</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col7" class="data row10 col7" >-0.74</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col8" class="data row10 col8" >-0.74</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col9" class="data row10 col9" >0.22</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col10" class="data row10 col10" >1</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow10_col11" class="data row10 col11" >-0.15</td>
            </tr>
            <tr>
                        <th id="T_278b9158_8673_11e9_a6f7_80c5f2b6775clevel0_row11" class="row_heading level0 row11" >CASH</th>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col0" class="data row11 col0" >0.16</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col1" class="data row11 col1" >-0.15</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col2" class="data row11 col2" >0.0016</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col3" class="data row11 col3" >0.0062</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col4" class="data row11 col4" >-0.12</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col5" class="data row11 col5" >-0.2</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col6" class="data row11 col6" >0.027</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col7" class="data row11 col7" >0.071</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col8" class="data row11 col8" >0.15</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col9" class="data row11 col9" >-0.045</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col10" class="data row11 col10" >-0.15</td>
                        <td id="T_278b9158_8673_11e9_a6f7_80c5f2b6775crow11_col11" class="data row11 col11" >1</td>
            </tr>
    </tbody></table>



Below you can see an average return(arithmetic), standard deviation by asset class and correlation and covariation matrix. Geometric returns can be used instead, but the difference is small anyway.


```python
mean_asset_rtn, std_asset_rtn = [annual_returns.mean(), annual_returns.std()]
#printing parameters
params = pd.DataFrame(columns=mean_asset_rtn.index, index = ['Mean_return','Standard_deviation'])
for key, rtn, stdev in zip(mean_asset_rtn.index, mean_asset_rtn, std_asset_rtn):
    params[key] = [f'{rtn*100:.02f}%', f'{stdev*100:.02f}%']
params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean_return</th>
      <td>4.68%</td>
      <td>7.78%</td>
      <td>6.21%</td>
      <td>5.60%</td>
      <td>11.16%</td>
      <td>6.72%</td>
      <td>6.34%</td>
      <td>-6.01%</td>
      <td>-2.46%</td>
      <td>6.55%</td>
      <td>12.08%</td>
      <td>2.46%</td>
    </tr>
    <tr>
      <th>Standard_deviation</th>
      <td>6.77%</td>
      <td>18.10%</td>
      <td>6.44%</td>
      <td>6.87%</td>
      <td>29.77%</td>
      <td>12.66%</td>
      <td>7.35%</td>
      <td>8.18%</td>
      <td>22.53%</td>
      <td>25.08%</td>
      <td>35.56%</td>
      <td>1.20%</td>
    </tr>
  </tbody>
</table>
</div>



It seems like equities are doing better than bonds, however equities are more volitile. Makes sense.
Let's take a look at correlation matrix


```python
covmat, corrmat = [annual_returns.cov(), annual_returns.corr()]
corrmat.style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col1 {
            background-color:  #b9d0f9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col2 {
            background-color:  #e0dbd8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col3 {
            background-color:  #f59d7e;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col4 {
            background-color:  #c6d6f1;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col5 {
            background-color:  #c4d5f3;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col6 {
            background-color:  #f7b093;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col7 {
            background-color:  #efcebd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col8 {
            background-color:  #f5c2aa;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col9 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col10 {
            background-color:  #86a9fc;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col11 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col0 {
            background-color:  #6f92f3;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col2 {
            background-color:  #d5dbe5;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col3 {
            background-color:  #efcebd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col4 {
            background-color:  #d44e41;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col5 {
            background-color:  #e16751;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col6 {
            background-color:  #e9d5cb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col7 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col8 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col9 {
            background-color:  #f4c6af;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col10 {
            background-color:  #de614d;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col11 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col0 {
            background-color:  #e3d9d3;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col1 {
            background-color:  #f3c7b1;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col3 {
            background-color:  #f29274;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col4 {
            background-color:  #f7ba9f;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col5 {
            background-color:  #f4987a;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col6 {
            background-color:  #f5a081;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col7 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col8 {
            background-color:  #85a8fc;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col9 {
            background-color:  #bfd3f6;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col10 {
            background-color:  #e6d7cf;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col11 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col0 {
            background-color:  #f59c7d;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col1 {
            background-color:  #f7af91;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col2 {
            background-color:  #f39475;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col4 {
            background-color:  #f7b194;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col5 {
            background-color:  #ee8468;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col6 {
            background-color:  #d44e41;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col7 {
            background-color:  #93b5fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col8 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col9 {
            background-color:  #ead4c8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col10 {
            background-color:  #e1dad6;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col11 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col0 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col1 {
            background-color:  #d44e41;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col2 {
            background-color:  #e4d9d2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col3 {
            background-color:  #edd1c2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col5 {
            background-color:  #e67259;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col6 {
            background-color:  #ecd3c5;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col7 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col8 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col9 {
            background-color:  #f6bda2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col10 {
            background-color:  #e57058;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col11 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col0 {
            background-color:  #93b5fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col1 {
            background-color:  #df634e;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col2 {
            background-color:  #f7b194;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col3 {
            background-color:  #f49a7b;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col4 {
            background-color:  #e36c55;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col6 {
            background-color:  #f7b194;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col7 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col8 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col9 {
            background-color:  #f7bca1;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col10 {
            background-color:  #f4987a;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col11 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col0 {
            background-color:  #f7af91;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col1 {
            background-color:  #f7b89c;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col2 {
            background-color:  #f6a283;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col3 {
            background-color:  #d55042;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col4 {
            background-color:  #f7b497;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col5 {
            background-color:  #f49a7b;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col7 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col8 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col9 {
            background-color:  #f3c8b2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col10 {
            background-color:  #dfdbd9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col11 {
            background-color:  #6f92f3;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col0 {
            background-color:  #cdd9ec;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col2 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col4 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col7 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col8 {
            background-color:  #eb7d62;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col9 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col10 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col11 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col0 {
            background-color:  #e6d7cf;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col1 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col2 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col3 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col4 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col5 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col6 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col7 {
            background-color:  #e9785d;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col8 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col9 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col10 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col11 {
            background-color:  #b9d0f9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col0 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col1 {
            background-color:  #f7af91;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col2 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col3 {
            background-color:  #e0dbd8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col4 {
            background-color:  #f6a586;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col5 {
            background-color:  #f7ac8e;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col6 {
            background-color:  #eed0c0;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col7 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col8 {
            background-color:  #81a4fb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col9 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col10 {
            background-color:  #f5c1a9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col11 {
            background-color:  #6e90f2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col1 {
            background-color:  #dd5f4b;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col2 {
            background-color:  #bad0f8;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col3 {
            background-color:  #b3cdfb;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col4 {
            background-color:  #e46e56;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col5 {
            background-color:  #f59d7e;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col6 {
            background-color:  #b1cbfc;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col7 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col8 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col9 {
            background-color:  #ebd3c6;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col10 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col11 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col0 {
            background-color:  #b1cbfc;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col1 {
            background-color:  #adc9fd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col2 {
            background-color:  #6f92f3;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col3 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col4 {
            background-color:  #b9d0f9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col5 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col6 {
            background-color:  #7b9ff9;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col7 {
            background-color:  #d6dce4;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col8 {
            background-color:  #e4d9d2;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col9 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col10 {
            background-color:  #aec9fc;
            color:  #000000;
        }    #T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col11 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775c" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >FI.DEV</th>        <th class="col_heading level0 col1" >EQ.DEV</th>        <th class="col_heading level0 col2" >FI.EM</th>        <th class="col_heading level0 col3" >FI.CORP</th>        <th class="col_heading level0 col4" >EQ.EM</th>        <th class="col_heading level0 col5" >FI.HY</th>        <th class="col_heading level0 col6" >FI.IL</th>        <th class="col_heading level0 col7" >HF</th>        <th class="col_heading level0 col8" >RE.SEC</th>        <th class="col_heading level0 col9" >COMMOD</th>        <th class="col_heading level0 col10" >Private EQ</th>        <th class="col_heading level0 col11" >CASH</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row0" class="row_heading level0 row0" >FI.DEV</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col0" class="data row0 col0" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col1" class="data row0 col1" >-0.15</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col2" class="data row0 col2" >0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col3" class="data row0 col3" >0.65</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col4" class="data row0 col4" >-0.071</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col5" class="data row0 col5" >-0.0069</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col6" class="data row0 col6" >0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col7" class="data row0 col7" >0.23</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col8" class="data row0 col8" >0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col9" class="data row0 col9" >-0.041</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col10" class="data row0 col10" >-0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow0_col11" class="data row0 col11" >0.11</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row1" class="row_heading level0 row1" >EQ.DEV</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col0" class="data row1 col0" >-0.15</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col1" class="data row1 col1" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col2" class="data row1 col2" >0.29</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col3" class="data row1 col3" >0.44</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col4" class="data row1 col4" >0.83</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col5" class="data row1 col5" >0.76</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col6" class="data row1 col6" >0.39</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col7" class="data row1 col7" >-0.84</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col8" class="data row1 col8" >-0.7</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col9" class="data row1 col9" >0.44</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col10" class="data row1 col10" >0.78</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow1_col11" class="data row1 col11" >-0.21</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row2" class="row_heading level0 row2" >FI.EM</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col0" class="data row2 col0" >0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col1" class="data row2 col1" >0.29</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col2" class="data row2 col2" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col3" class="data row2 col3" >0.69</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col4" class="data row2 col4" >0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col5" class="data row2 col5" >0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col6" class="data row2 col6" >0.64</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col7" class="data row2 col7" >-0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col8" class="data row2 col8" >-0.32</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col9" class="data row2 col9" >0.1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col10" class="data row2 col10" >0.17</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow2_col11" class="data row2 col11" >-0.11</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row3" class="row_heading level0 row3" >FI.CORP</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col0" class="data row3 col0" >0.65</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col1" class="data row3 col1" >0.44</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col2" class="data row3 col2" >0.69</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col3" class="data row3 col3" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col4" class="data row3 col4" >0.42</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col5" class="data row3 col5" >0.66</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col6" class="data row3 col6" >0.87</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col7" class="data row3 col7" >-0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col8" class="data row3 col8" >-0.21</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col9" class="data row3 col9" >0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col10" class="data row3 col10" >0.14</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow3_col11" class="data row3 col11" >-0.099</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row4" class="row_heading level0 row4" >EQ.EM</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col0" class="data row4 col0" >-0.071</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col1" class="data row4 col1" >0.83</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col2" class="data row4 col2" >0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col3" class="data row4 col3" >0.42</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col4" class="data row4 col4" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col5" class="data row4 col5" >0.72</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col6" class="data row4 col6" >0.41</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col7" class="data row4 col7" >-0.84</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col8" class="data row4 col8" >-0.62</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col9" class="data row4 col9" >0.48</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col10" class="data row4 col10" >0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow4_col11" class="data row4 col11" >-0.15</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row5" class="row_heading level0 row5" >FI.HY</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col0" class="data row5 col0" >-0.0069</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col1" class="data row5 col1" >0.76</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col2" class="data row5 col2" >0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col3" class="data row5 col3" >0.66</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col4" class="data row5 col4" >0.72</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col5" class="data row5 col5" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col6" class="data row5 col6" >0.57</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col7" class="data row5 col7" >-0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col8" class="data row5 col8" >-0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col9" class="data row5 col9" >0.49</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col10" class="data row5 col10" >0.56</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow5_col11" class="data row5 col11" >-0.3</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row6" class="row_heading level0 row6" >FI.IL</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col0" class="data row6 col0" >0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col1" class="data row6 col1" >0.39</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col2" class="data row6 col2" >0.64</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col3" class="data row6 col3" >0.87</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col4" class="data row6 col4" >0.41</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col5" class="data row6 col5" >0.57</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col6" class="data row6 col6" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col7" class="data row6 col7" >-0.36</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col8" class="data row6 col8" >-0.21</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col9" class="data row6 col9" >0.42</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col10" class="data row6 col10" >0.12</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow6_col11" class="data row6 col11" >-0.086</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row7" class="row_heading level0 row7" >HF</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col0" class="data row7 col0" >0.23</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col1" class="data row7 col1" >-0.84</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col2" class="data row7 col2" >-0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col3" class="data row7 col3" >-0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col4" class="data row7 col4" >-0.84</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col5" class="data row7 col5" >-0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col6" class="data row7 col6" >-0.36</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col7" class="data row7 col7" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col8" class="data row7 col8" >0.68</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col9" class="data row7 col9" >-0.48</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col10" class="data row7 col10" >-0.79</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow7_col11" class="data row7 col11" >0.034</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row8" class="row_heading level0 row8" >RE.SEC</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col0" class="data row8 col0" >0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col1" class="data row8 col1" >-0.7</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col2" class="data row8 col2" >-0.32</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col3" class="data row8 col3" >-0.21</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col4" class="data row8 col4" >-0.62</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col5" class="data row8 col5" >-0.58</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col6" class="data row8 col6" >-0.21</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col7" class="data row8 col7" >0.68</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col8" class="data row8 col8" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col9" class="data row8 col9" >-0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col10" class="data row8 col10" >-0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow8_col11" class="data row8 col11" >0.19</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row9" class="row_heading level0 row9" >COMMOD</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col0" class="data row9 col0" >-0.041</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col1" class="data row9 col1" >0.44</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col2" class="data row9 col2" >0.1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col3" class="data row9 col3" >0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col4" class="data row9 col4" >0.48</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col5" class="data row9 col5" >0.49</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col6" class="data row9 col6" >0.42</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col7" class="data row9 col7" >-0.48</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col8" class="data row9 col8" >-0.34</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col9" class="data row9 col9" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col10" class="data row9 col10" >0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow9_col11" class="data row9 col11" >-0.091</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row10" class="row_heading level0 row10" >Private EQ</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col0" class="data row10 col0" >-0.37</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col1" class="data row10 col1" >0.78</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col2" class="data row10 col2" >0.17</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col3" class="data row10 col3" >0.14</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col4" class="data row10 col4" >0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col5" class="data row10 col5" >0.56</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col6" class="data row10 col6" >0.12</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col7" class="data row10 col7" >-0.79</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col8" class="data row10 col8" >-0.71</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col9" class="data row10 col9" >0.35</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col10" class="data row10 col10" >1</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow10_col11" class="data row10 col11" >-0.17</td>
            </tr>
            <tr>
                        <th id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775clevel0_row11" class="row_heading level0 row11" >CASH</th>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col0" class="data row11 col0" >0.11</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col1" class="data row11 col1" >-0.21</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col2" class="data row11 col2" >-0.11</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col3" class="data row11 col3" >-0.099</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col4" class="data row11 col4" >-0.15</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col5" class="data row11 col5" >-0.3</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col6" class="data row11 col6" >-0.086</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col7" class="data row11 col7" >0.034</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col8" class="data row11 col8" >0.19</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col9" class="data row11 col9" >-0.091</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col10" class="data row11 col10" >-0.17</td>
                        <td id="T_27a0bea4_8673_11e9_baf1_80c5f2b6775crow11_col11" class="data row11 col11" >1</td>
            </tr>
    </tbody></table>



Intuitively, high correlation between assets - a bad thing, low correlation - a good thing. 


```python
def __solve_portfolio(S, pbar, G, h, A, b, pmultiplier):
    '''Solves quadratic problem and returns weights and correcpondong return and risk
    '''
    x = solvers.qp(S, -pmultiplier*pbar, G, h, A=A, b=b)['x']
    return np.array(x), blas.dot(pbar, x), np.sqrt(blas.dot(x, S*x))

def efficient_frontier(rtns, num_eff_ports = 15, min_weights_vect=None, max_weights_vect=None):
    '''Returns [array of weights, return, volatility]
    you can specify min. and max weights for simulation, like min_weights_vect = [0, .10, ..., -.20] 
    '''
    n = rtns.shape[1]
    #calculate returns and covariance
    S = opt.matrix(np.matrix(rtns.cov()))
    pbar = opt.matrix(np.mean(rtns))
    
    #by default an asset weight is from 0 to 100% (no leverage, no borrowing)
    if min_weights_vect is None: min_weights_vect=[0]*n 
    if max_weights_vect is None: max_weights_vect=[1]*n 
        
    
    # Create constraint matrices
    G = None
    h = None
    if not min_weights_vect is None:
        G = -np.eye(n)
        h = np.matrix(min_weights_vect)
    if not max_weights_vect is None:
        G = np.concatenate((G,np.eye(n)))
        h = np.concatenate((h, np.matrix(max_weights_vect)), axis=1)#recheck that weights are set up correctly
    if (not min_weights_vect is None) or (not max_weights_vect is None):
        G = opt.matrix(G)   # negative n x n identity matrix
        h = opt.matrix(h, (h.shape[1],1), 'd')
    #we can add group counstraints later if required
    
    A = opt.matrix(1.0, (1, n)) #sum of weights should be equal to 1
    b = opt.matrix(1.0)
    
    minVarPort = __solve_portfolio(S, pbar, G, h, A, b, 0)
    maxRtnPort = __solve_portfolio(S, pbar, G, h, A, b, 10E5)
    
    target_returns = np.linspace(minVarPort[1], maxRtnPort[1], num_eff_ports-1, endpoint=False)
    
    frontier = list()
    frontier.append(minVarPort)
    for target_rtn in target_returns[1:]:
        port = __solve_portfolio(S, 
                                 pbar, 
                                 G, 
                                 h, 
                                 opt.matrix(np.concatenate((np.matrix([1]*n),np.transpose(pbar)))), #adjust Amat to add return target
                                 opt.matrix([1.0, target_rtn], (2,1)), #adjust bvac to add return target
                                 0
                                )
        frontier.append(port)
    frontier.append(maxRtnPort)

    return frontier
```

Wow, all is set up to find our efficient frontier and earn billions of dollars. 
We'll use annualised returns, because compuunding effect is taken into account.


```python
front = efficient_frontier(annual_returns)
```


```python
plt.figure(figsize=(12, 6))
plt.plot([std for _,_,std in front], [rtn for _, rtn, _ in front],'y-o')
plt.title('Efficient frontier')
plt.xlabel('Volatility, %')
plt.ylabel('Return, %')
plt.gca().set_yticklabels(['{:.00f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_xticklabels(['{:.00f}%'.format(x*100) for x in plt.gca().get_xticks()])
plt.grid(True)
```


![png](output_37_0.png)



```python
wt = np.concatenate([w for w, _, _ in front], axis=1)
plt.figure(figsize=(12, 6))
plt.title('Weights of assets in eficient portfolios')
plt.stackplot([std for _ , _,std in front], wt, labels = annual_returns.columns, alpha=.7)
plt.gca().set_yticklabels(['{:.00f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_xticklabels(['{:.00f}%'.format(x*100) for x in plt.gca().get_xticks()])
plt.legend(loc='upper right')
plt.ylabel('Proportion of assets, %')
plt.xlabel('Volatility, %')
plt.grid(True)
```


![png](output_38_0.png)



```python
front_df = pd.DataFrame(np.transpose(np.concatenate([w for w, _, _ in front], axis=1)), 
                        [std for _ , _,std in front], 
                        columns=annual_returns.columns)
front_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FI.DEV</th>
      <th>EQ.DEV</th>
      <th>FI.EM</th>
      <th>FI.CORP</th>
      <th>EQ.EM</th>
      <th>FI.HY</th>
      <th>FI.IL</th>
      <th>HF</th>
      <th>RE.SEC</th>
      <th>COMMOD</th>
      <th>Private EQ</th>
      <th>CASH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.84%</th>
      <td>0.00%</td>
      <td>2.31%</td>
      <td>2.16%</td>
      <td>0.00%</td>
      <td>0.64%</td>
      <td>3.35%</td>
      <td>0.02%</td>
      <td>15.93%</td>
      <td>0.21%</td>
      <td>0.37%</td>
      <td>1.34%</td>
      <td>73.67%</td>
    </tr>
    <tr>
      <th>0.96%</th>
      <td>0.07%</td>
      <td>1.34%</td>
      <td>4.77%</td>
      <td>0.02%</td>
      <td>0.00%</td>
      <td>2.55%</td>
      <td>0.71%</td>
      <td>6.14%</td>
      <td>0.80%</td>
      <td>0.16%</td>
      <td>1.01%</td>
      <td>82.42%</td>
    </tr>
    <tr>
      <th>1.32%</th>
      <td>1.69%</td>
      <td>0.48%</td>
      <td>10.56%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.18%</td>
      <td>3.47%</td>
      <td>0.00%</td>
      <td>1.27%</td>
      <td>0.06%</td>
      <td>1.43%</td>
      <td>80.87%</td>
    </tr>
    <tr>
      <th>2.19%</th>
      <td>6.68%</td>
      <td>0.00%</td>
      <td>20.77%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>7.36%</td>
      <td>0.00%</td>
      <td>1.78%</td>
      <td>0.01%</td>
      <td>3.09%</td>
      <td>60.31%</td>
    </tr>
    <tr>
      <th>3.26%</th>
      <td>11.82%</td>
      <td>0.00%</td>
      <td>31.14%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>10.63%</td>
      <td>0.00%</td>
      <td>2.41%</td>
      <td>0.00%</td>
      <td>4.60%</td>
      <td>39.40%</td>
    </tr>
    <tr>
      <th>4.38%</th>
      <td>17.00%</td>
      <td>0.00%</td>
      <td>41.54%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>13.83%</td>
      <td>0.00%</td>
      <td>3.03%</td>
      <td>0.02%</td>
      <td>6.10%</td>
      <td>18.47%</td>
    </tr>
    <tr>
      <th>5.53%</th>
      <td>21.26%</td>
      <td>0.00%</td>
      <td>50.99%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>17.88%</td>
      <td>0.00%</td>
      <td>2.55%</td>
      <td>0.01%</td>
      <td>7.30%</td>
      <td>0.01%</td>
    </tr>
    <tr>
      <th>7.12%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>51.21%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>38.58%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>10.21%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>9.96%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>36.51%</td>
      <td>0.00%</td>
      <td>7.56%</td>
      <td>0.00%</td>
      <td>39.40%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>16.53%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>13.41%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>22.80%</td>
      <td>0.00%</td>
      <td>16.41%</td>
      <td>0.00%</td>
      <td>38.99%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>21.79%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>17.10%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>9.09%</td>
      <td>0.00%</td>
      <td>25.26%</td>
      <td>0.00%</td>
      <td>38.59%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>27.06%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>20.90%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>34.37%</td>
      <td>0.00%</td>
      <td>33.43%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>32.20%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>24.78%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>44.02%</td>
      <td>0.00%</td>
      <td>18.88%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>37.10%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>28.72%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>53.67%</td>
      <td>0.00%</td>
      <td>4.33%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>41.99%</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>35.56%</th>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>-0.00%</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>100.00%</td>
      <td>0.00%</td>
    </tr>
  </tbody>
</table>
</div>



So, if you don't like risk - CASH is your choise. If you're ready to take risks to achieve high return - invest in private equities. In-between - diversify your portfolio.

Let's look at possible return paths of the portfolio №8.


```python
port_num = 8
front_df.iloc[port_num-1]
```




    FI.DEV        0.00%
    EQ.DEV        0.00%
    FI.EM        51.21%
    FI.CORP       0.00%
    EQ.EM         0.00%
    FI.HY         0.00%
    FI.IL        38.58%
    HF            0.00%
    RE.SEC        0.00%
    COMMOD        0.00%
    Private EQ   10.21%
    CASH          0.00%
    Name: 0.07117887656378721, dtype: float64




```python
print(f'Expected annual return: {front[port_num-1][1]:.02%}, Volatility: {front[port_num-1][2]:.02%}')
```

    Expected annual return: 6.86%, Volatility: 7.12%
    

Of course, past performance is no guarantee of future results, and these days we have to adjust returns downward (not in this example).


```python
years = 5
simulations = 40
eff_port_rtns = montecarlo(np.dot(returns, front[port_num-1][0]), num_simulations=years*simulations*12)
```

Plotting possible return paths of the portfolio


```python
eff_port_rtns = np.array(eff_port_rtns).reshape(years*12,simulations)#find returns of a portfolio generated using prescribed weights
eff_port_rtns = pd.DataFrame(eff_port_rtns)
cum_ret_sim = cumulative_returns(eff_port_rtns)
cum_ret_sim = cum_ret_sim.set_index( pd.date_range(start=returns.index[-1], periods = len(cum_ret_sim), freq="M") )
plot_returns(cum_ret_sim, legend = False)
```


    <Figure size 432x288 with 0 Axes>



![png](output_47_1.png)


# Conclusion

In this worksheet we have calculated returns of a range of asset classes, its parameters like mean, standard deviation, historical and variance-covariance VaR and Expected shortfall, calculated Markovitz efficient portfolios, simulated returns based on Bootstrap and Monte-Carlo methods, and simulated expected returns of an efficient portfolio.
