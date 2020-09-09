# Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process
 A model simulation shows how pairs trading could be used for two S&P500 traded stocks. It proofs that the strategy is successful on real data, which is downloaded via the
 [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/). In the report and poster section a profound analysis about the pairs trading underlying Ornstein-Uhlenbeck process is done. Rigorous mathematical formulations can be found in this section also. 
 
 
 Keywords: Stochastic differential equations, Ornstein-Ulenbeck Process, Pairs Trading, S&P500 

### Work Flow & Manual: 
1.  Load stock data from [slib.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/slib.py) with **stockvals()**.
2.  Determine two stocks suitable for the pairs trading strategy with **find_cointegrated_pairs()** in [CointHurst.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/CointHurst.py). Note tha the output 'pairs' already suggest pairs of stocks with a significance level below 0.05 using a cointegration test. A full map of pvalues for pairwise cointegration can be displayed with **covplot()**.
3.  Use **pairstrade()** in [PairsTrade.py](https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Code/PairsTrade.py) to execute the pairs trading strategy on the two stocks (id0, id1) under consideration. The parameters that determine the success of the strategy are defined in the **pairstrade()** input *param: f8[d,buy_signal,sell_signal]* - threshold parameters for the time window, buying and selling signal. A graph that shows the buying and selling signal and how much money the strategy would have made can be displayed by setting *plot = 'True'*.

### Getting started

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import methods1lib as lb1
import mclib as mclb
import mpllib as mplb

#%% Load data
cell_data = pd.read_csv("cellData.csv",delimiter= ",")
cell_data = cell_data.to_numpy(); cell_data = cell_data.astype(float) 

#%% Set Parameters
#--Simulation specific Parameters
patient = 4
depiction = 'General' #'Detailed'; 'Grained'; 
cellTypes_orig = [len(cellTypes_General)]
limiter = False
runs = 1 #number of optimisations for every sample

#--MCMC Encoding Parameters
N = 150 #MC Epochs
boxlength = 780. #micro meter
TT_enc = np.array([1.]); Tsteps = len(TT_enc) #MC Temperature
#--MPL Decoding Parameters
TT_dec = TT_enc #MPL decoding temperature
```

### Map of pairwise cointegrated stocks
 
  Cointegration  
:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/cointegration.png" width="400" height="350" />  
 
Setting the parameter depiction to 'General', 'Grained', 'Detaild' processes the data to a pathological image with finer subdivision of different cell types.

### Buying-Selling Signal & Trading Performance
 
  Buying-Selling Signal |   Trade Performance
:-------------------------:|:-------------------------:
 <img align="center" src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/buySell.png" width="500" height="400" />  |  <img src="https://github.com/david-alber/Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process/blob/master/Images/trade_performance.png" width="400" height="320" />
 
A tissue sample of the tumor environment of patient 4 (compartemntalized type) is analysed: The image data is used to infer a parameter set of inter cellular connectivities that most likely is responsible for guiding the pattern formation at hand. The infered parameterset is the imput for the tumor generating model, which simulates a tumor environment, according to the previous analysis of maximum pseudo likelihood inference.



### How to contribute
Fork from the `Developer`- branch and pull request to merge back into the original `Developer`- branch. 
Working updates and improvements will then be merged into the `Master` branch, which will always contain the latest working version.

With: 
* [Jean Hausser](https://www.scilifelab.se/researchers/jean-hausser/)

### Dependencies
 [Numba](https://numba.pydata.org/), 
 [scikit-learn](https://scikit-learn.org/stable/), 
 [Numpy](https://numpy.org/), 
 [Scipy](https://www.scipy.org/), 
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/)
 
