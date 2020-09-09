import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import slib as slb
plt.close('all')
#%%
start_year = 2010; end_year = 2020;
end_yearTe = end_year
names = np.load('names.npy'); symbols = np.load('symbols.npy'); 
variation = np.load('variation.npy'); close_val = np.load('close_val.npy'); open_val = np.load('open_val.npy')
pairs = np.load('pairs.npy'); pvalues = np.load('pvalues.npy'); scores = np.load('scores.npy'); 

data = pd.DataFrame.from_records(close_val)
days = np.shape(data)[1]
train = 0.7
data_train = data.loc[:,0:int(days*train)]
data_test =  data.loc[:,int(days*train):]
end_year = int(start_year + (end_year-start_year)*train) #redifine end year according to training samples
start_yearTe = end_year;
id0 = pairs[1,0]; id1 = pairs[1,1]

#%%
def returnMaximizer(random_walks,function,n_param,data,id0,id1,start_year,end_year,d):
    """
    random_walks : f8; number of random walks, i.e different random init guesses
    function : function handle; function to optimize: sp.optimize.minimize(...,method='Nelder-Mead')
    n_param : f8; number of free parameters to optimize

    Returns: opt_object: .x->optimal parameters; .fun-> fct evaluation at opt parameters
    """
    #Funcition specific definitions
    d = int(d)
    
    minimum = 9e15
    for i in tqdm(range(0,random_walks),position=0,desc ='MPL Maximizer'): #Minimize for different initial guesses
        initial_guess = np.zeros(n_param)
        initial_guess[0] = np.random.uniform(0,3);
        initial_guess[1] = np.random.uniform(0,initial_guess[0])#print(initial_guess) #init guess for values of connectivity matrix Jmat
        optimizer_object = sp.optimize.minimize(function,initial_guess,method='Nelder-Mead')
        y = optimizer_object.fun
        if (y < minimum): #keep parameters if -MPL is smaller than before
            minimum = y;# print(minimum)
            opt_object = optimizer_object;         
    print(f'Parameters that minimize returns: buy,sell = {opt_object.x}')
    return opt_object 

def pairstrade_maxi(param):
    buy_signal,sell_signal = param[0],param[1] 
    S0 = data.iloc[id0][:]; S1 = data.iloc[id1][:]
    
    k = (S0/S1).rolling(window=int(d), center=False).mean()
    Y_tilde = np.log(S0/(k*S1))[d:-1] # model implied OU value based on deviation from k_old
    
    #TRADE THE PAIR BASED ON OU IMPLIED OVER-/UNDERVALUATION.
    t = np.linspace(start_year,end_year,int(len(S0)))
    S0=S0[d:-1]; S1=S1[d:-1]; k = k[d:-1]; t = t[d:-1]
  
    money = np.zeros(len(Y_tilde)) # start trading with no positions
    S0_held = np.zeros(len(Y_tilde))
    S1_held = np.zeros(len(Y_tilde))
    
    buy = []; sell = []
    for i in range(len(S0)):
        if  Y_tilde.iloc[i] >  buy_signal:            # short the pair when OU up
            S0_held[i] = S0_held[i-1] - 1
            S1_held[i] = S1_held[i-1] + S0.iloc[i]/S1.iloc[i]
            money[i] = money[i-1] - 0.005 * 2 * S0.iloc[i]  # 50 bps tx costs
            
            buy.append((t[i], S1.iloc[i]))
            sell.append((t[i], S0.iloc[i]))
            
        elif Y_tilde.iloc[i] < -buy_signal:           # long the pair when OU down
            S0_held[i] = S0_held[i-1] + 1
            S1_held[i] = S1_held[i-1] - S0.iloc[i]/S1.iloc[i]
            money[i] = money[i-1] - 0.005 * 2 * S0.iloc[i]
            
            buy.append((t[i], S0.iloc[i]))
            sell.append((t[i], S1.iloc[i]))
            
        elif abs(Y_tilde.iloc[i]) < sell_signal and S0_held[i-1]!=0:     # clear position when ratio normal
            money[i] = money[i-1] + (S0_held[i-1]*S0.iloc[i] + S1_held[i-1]*S1.iloc[i]) * 0.995 
            S0_held[i], S1_held[i] = 0, 0
            
            if S0_held[i-1] < 0:
                buy.append((t[i], S0.iloc[i]))
                sell.append((t[i], S1.iloc[i]))
            else: 
                buy.append((t[i], S1.iloc[i]))
                sell.append((t[i], S0.iloc[i]))
                
        else:
            money[i] = money[i-1]
            S0_held[i] = S0_held[i-1]
            S1_held[i] = S1_held[i-1]
    ret = 9e10-(money[-1]); 
    return ret #flip sign for optimizer

d = 252
n_param = 2
random_walks = 20
opt_object = returnMaximizer(random_walks,pairstrade_maxi,n_param,data,id0,id1,start_year,end_year,d)

#%%
buy_signal, sell_signal = opt_object.x[0], opt_object.x[1]
buy_signal, sell_signal = 0.05, 0.01

windows = np.linspace(10,700,100)
train_return = np.zeros(len(windows)); test_return = np.zeros(len(windows))
for i in range(0,len(windows)):
    d = int(windows[i])
    param = np.array([d,buy_signal,sell_signal]) #d,buy_signal,sell_signal
    #training data evaluation
    moneyTr,_,_,_,_,_,_,_ = slb.pairstrade(param,data_train,id0,id1,names,start_year,end_year,plot=False)                                                                 
    #test data evaluation
    moneyTe,_,_,_,_,_,_,_ = slb.pairstrade(param,data_test,id0,id1,names,start_yearTe,end_yearTe,plot=False)
    
    train_return[i] = moneyTr[-1]
    test_return[i] = moneyTe[-1]
    
#%% Plot Window Fit
plt.figure('FittingWindow')
plt.plot(windows,train_return,color='g',label='Trainin data')
plt.plot(windows,test_return,color='r',label='Test data')
plt.title('Window analysis')
plt.xlabel('Window length [days]'); plt.ylabel('Final money [EUR]')
plt.legend(); plt.show()
#%%
buy_signal, sell_signal = opt_object.x[0], opt_object.x[1]
best_window = windows[np.argmax(train_return)]
param = np.array([best_window,buy_signal,sell_signal])
moneybest,_,_,_,_,_,_,_ = slb.pairstrade(param,data_train,id0,id1,names,start_year,end_year,plot=True)
#%%
best_window=252
buy_signal, sell_signal = 0.05, 0.01
param = np.array([best_window,buy_signal,sell_signal])
moneybest,_,_,_,_,_,_,_ = slb.pairstrade(param,data_train,id0,id1,names,start_year,end_year,plot=True)