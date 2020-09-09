# Pairs-Trading-as-application-to-the-Ornstein-Uhlenbeck-Process
 A model simulation shows how pairs trading could be used for two S&P-500 traded stocks. It proofs that the strategy is successful on real data, which is downloaded via the
 [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/). In the report and poster section a profound analysis about the pairs trading underlying Ornstein-Uhlenbeck process is done. Rigorous mathematical formulations can be found in this section also. 
 
 
 Keywords: Stochastic differential equations, Ornstein-Ulenbeck Process, Pairs Trading, S\&P500 

### Work flow: 
1.  Load cellData 
2.  Set parameters for: maximum pseudo likelihood parameter inference and generating MCMC simulation
3.  Infere parameters from image with MPL: paramInference()
4.  Simulate with infered connectivity matrix: mcGenerator()


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

### Cell Type Depictions
 
  General | Grained | Detailed  
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Parameter-Learning-In-Tumor-Environments/blob/master/Images/p12ComGeneral.png" width="400" height="350" />  |  <img src="https://github.com/david-alber/Parameter-Learning-In-Tumor-Environments/blob/master/Images/p12ComGrained.png" width="400" height="350" /> |  <img src="https://github.com/david-alber/Parameter-Learning-In-Tumor-Environments/blob/master/Images/p12ComDetailed.png" width="400" height="350" />  
 
Setting the parameter depiction to 'General', 'Grained', 'Detaild' processes the data to a pathological image with finer subdivision of different cell types.

### Pathological Image - Simulated Configuration
 
  Pathological Image |   Simulated Configuration
:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Parameter-Learning-In-Tumor-Environments/blob/master/Images/pathoP4General.png" width="400" height="320" />  |  <img src="https://github.com/david-alber/Parameter-Learning-In-Tumor-Environments/blob/master/Images/configP4General.png" width="400" height="320" />
 
A tissue sample of the tumor environment of patient 4 (compartemntalized type) is analysed: The image data is used to infer a parameter set of inter cellular connectivities that most likely is responsible for guiding the pattern formation at hand. The infered parameterset is the imput for the tumor generating model, which simulates a tumor environment, according to the previous analysis of maximum pseudo likelihood inference.

### Comments
For legal reasons the data files, which were used for the analysis and the simulations, cannot be shared publically. Please get in touch with me if you are interested in the data that drives the presented results.


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
 
