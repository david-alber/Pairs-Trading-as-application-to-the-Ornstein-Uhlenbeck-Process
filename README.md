# Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process
 A model simulation shows how pairs trading could be used for two S&P500 traded stocks. It proofs that the strategy is successful on real data, which is downloaded via the
 [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/). In the report and poster section a profound analysis about the pairs trading underlying Ornstein-Uhlenbeck process is done. Rigorous mathematical formulations can be found in this section also. 
 
 Keywords: Stochastic differential equations, Ornstein-Ulenbeck Process, Pairs Trading, S&P500 

### Work Flow & Manual: 
1.  Load stock data from [slib.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/slib.py) with **stockvals()**.
2.  Determine two stocks suitable for the pairs trading strategy with **find_cointegrated_pairs()** in [CointHurst.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/CointHurst.py). Note tha the output 'pairs' already suggest pairs of stocks with a significance level below 0.05 using a cointegration test. A full map of pvalues for pairwise cointegration can be displayed with **covplot()**.
3.  Use **pairstrade()** in [PairsTrade.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/PairsTrade.py) to execute the pairs trading strategy on the two stocks (id0, id1) under consideration. The parameters that determine the success of the strategy are defined in the **pairstrade()** input *param: f8[d,buy_signal,sell_signal]* - threshold parameters for the time window, buying and selling signal. A graph that shows the buying and selling signal and how much money the strategy would have made can be displayed by setting *plot = 'True'*.


### Map of pairwise cointegrated stocks
 
  Cointegration  
:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/cointegration.png" width="450" height="350" />  
 


### Buying-Selling Signal & Trading Performance
 
  Buying-Selling Signal |   Trade Performance
:-------------------------:|:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/buySell.png" width="400" height="320" />  |  <img src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/trade_performance.png" width="400" height="320" />
 




### How to contribute
Fork from the `Developer`- branch and pull request to merge back into the original `Developer`- branch. 
Working updates and improvements will then be merged into the `Master` branch, which will always contain the latest working version.

With: 
* [Daniel Brus](https://www.linkedin.com/in/daniel-brus)

### Dependencies 
 [scikit-learn](https://scikit-learn.org/stable/), 
 [Numpy](https://numpy.org/),  
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/), [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/),
 [slib (custom)](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/slib.py)
 
