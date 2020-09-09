# Pairs-Trading-as-application-of-the-Ornstein-Uhlenbeck-Process
 A model simulation shows how pairs trading could be used for two S&P500 traded stocks. It proofs that the strategy is successful on real data, which is downloaded via the
 [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/). In the [report and poster section](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/tree/master/Report%2BPoster) a profound analysis of the pairs trading underlying Ornstein-Uhlenbeck process is done. Rigorous mathematical formulations can be found in this section also. 
 
 Keywords: Stochastic differential equations, Ornstein-Ulenbeck Process, Pairs Trading, S&P500 

### Work Flow & Manual: 
1.  Load stock data from [slib.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/slib.py) with **stockvals()**.
2.  Determine two stocks suitable for the pairs trading strategy with **find_cointegrated_pairs()** in [CointHurst.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/CointHurst.py). Note tha the output 'pairs' already suggest pairs of stocks with a significance level below 0.05 using a cointegration test. A full map of pvalues for pairwise cointegration can be displayed with **covplot()**.
3.  Use **pairstrade()** in [PairsTrade.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/PairsTrade.py) to execute the pairs trading strategy on the two stocks (id0, id1) under consideration. The parameters that determine the success of the strategy are defined in the **pairstrade()** input *param: f8[d,buy_signal,sell_signal]* - threshold parameters for the time window, buying and selling signal. A graph that shows the buying and selling signal and how much money the strategy would have made can be displayed by setting *plot = True*.

### Getting started
```python
import pandas_datareader.data as web
from statsmodels.tsa.stattools import coint
import numpy as np
import matplotlib.pyplot as plt
import slib as slb #Custom library

#%% Load data
names[indices.astype(int)],symbols[indices.astype(int)],variation,close_val,open_val = stockvals(df,start_date,end_date)
data = pd.DataFrame.from_records(close_val)

#%% Find suitable stocks for pairs trading
scores, pvalues, pairs = slb.find_cointegrated_pairs(close_val)
slb.covplot(pvalues,symbols,'Cointegration')

#%% Perform trading strategy 
start_year = 2010; end_year = 2020;
d, buy_signal, sell_signal = 252 0.05, 0.01
param = [d,buy_signal,sell_signal]

 money,buy,sell,k,Y_tilde,t,S0,S1 = slb.pairstrade(param,data_train,id0,id1,names,start_year,end_year,plot=True)
```


### Map of pairwise cointegrated stocks
 
  Cointegration  
:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/cointegration.png" width="450" height="350" />  
 
 Map of p-values for pairwise cointegration. The statistical test for cointegration checks how likely the closing values of two stocks are mean reversive. The null hypothesis is no cointegration. Hence low p-values suggest that the stochastic processes under consideration show the property of being mean reversiv and can be described as an Ornstein-Uhlenbeck processes. They are thus suitable to perform the pairs trading strategy. 
 


### Buying-Selling Signal & Trading Performance
 
  Buying-Selling Signal |   Trade Performance
:-------------------------:|:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/buySell.png" width="400" height="320" />  |  <img src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/trade_performance.png" width="400" height="320" />
 
Buying and selling signals for two suitable S&P500 stocks (Adobe Inc.-Cintas Corporation) over a time period of 10 years (2010-2020).
The strategy started with 0 EUR, ended with 3000 EUR of hypothetical profit.



### How to contribute
Fork from the `Developer`- branch and pull request to merge back into the original `Developer`- branch. 
Working updates and improvements will then be merged into the `Master` branch, which will always contain the latest working version.

With: 
* [Daniel Brus](https://www.linkedin.com/in/daniel-brus)

### Dependencies 
 [Numpy](https://numpy.org/),  
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/), [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/), [Statsmodels](https://www.statsmodels.org/stable/index.html),
 [slib (custom)](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/slib.py)
 
