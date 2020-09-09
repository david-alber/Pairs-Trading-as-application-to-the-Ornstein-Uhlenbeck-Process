# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:37:42 2020

@author: alber
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold, metrics
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from tqdm import tqdm #Progressbar

from statsmodels.tsa.stattools import coint

def stockvals(df,start_date,end_date):
    """
    Converts opening/closing values of stocks from a pd dataframe to np arrays 
    of floats and converts the names to arrays of strings 
    IN:
    df : pandas dataframe of stocks
    start_date, end_date : datetime; start and end date
    end_date : TYPE
    RETURNS:
    names,symbols : str, stock names and symbols
    variation : f8[n_samples,n_timesteps], variation value between opening and closing price
    """
    #convert pd dataframes to strings
    symbols, names = df.Symbol, df.Security
    symbols = symbols.to_numpy()
    symbols = symbols.astype(str)
    names = names.to_numpy()
    names = names.astype(str)
    start_date_int = datetime_to_integer(start_date)
    #Stocks under consideration (from S&P500)
    n_stocks = len(symbols)
    #Open - Closing value of stocks (as float)
    indices = []; open_val = []; close_val = []
    for j in tqdm(range(0,n_stocks),position=0,desc='Loading Stock Data'):
        if j == 91:
            continue
        date_string=(df.iloc[j][6]).replace('-',''); #print(date_string)
        date_added = int(date_string[:8])
        if(date_added <= start_date_int):
            index = j
            indices = np.append(indices,index)
            quotes = web.DataReader(symbols[j], 'yahoo', start_date, end_date)
            opening = quotes.Open
            closing = quotes.Close
            open_val = np.append(open_val,opening,axis=0)
            close_val = np.append(close_val,closing,axis=0)
    open_val = open_val.reshape(len(indices),-1)
    close_val = close_val.reshape(len(indices),-1)
    variation = open_val-close_val
    return names[indices.astype(int)],symbols[indices.astype(int)],variation,close_val,open_val

def datetime_to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def blocking(tt,vector,blocks):
        """
        This is a function which helps to process big data files more easily
        by the method of block averaging. 
        For this the second argument is a vector with data, e.g. averaged temperature/stock price
        the second argument is another vector, e.g. time grid. 
        The third argument should be the number of blocks. 
        The more blocks, the more data points are taken into consideration. 
        If less blocks, more averaging takes place.
        Out: blockvec - blockaveraged vector
             blocktt - timesteps acording to blockaveraged data
             ...
             bdata - number of data points combined in one block
        """
        blockvec = np.zeros(blocks)
        elements = len(vector) 
        rest     = elements % blocks
        if rest != 0: #truncate vector if number of blocks dont fit in vector
            vector   = vector[0:-rest]
            tt       = tt[0:-rest]
            elements = len(vector)   
        meanA  = np.mean(vector)        
        bdata  = int((elements/blocks))#how many points per block
        sigBsq = 0; 
        for k in range(0,blocks):
            blockvec[k] = np.average(vector[k*bdata : (k+1)*bdata]) 
            sigBsq      = sigBsq + (blockvec[k]-meanA)**2    
        sigBsq *= 1/(blocks-1); 
        sigmaB = np.sqrt(sigBsq)
        error  = 1/np.sqrt(blocks)*sigmaB
        blocktt = tt[0:-1:bdata]
        return(blockvec,blocktt,error,sigmaB,bdata)    
    
def covplot(cov,symbols,title):
    """
    Plots a covariance matrix cov as heatmap. The x-,y tick lables are the input symbols
    cov : f8[n_samples,n_samples]; covariance matrix
    symbols : str[n_samples]; name of x and y tick labels
    """
    fig, ax = plt.subplots(figsize=(12,7))
    im = ax.imshow(cov)
    plt.colorbar(im, spacing='proportional')
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")
    
    ax.set_title(title)
    fig.tight_layout()
    plt.show()    
    
def stockplot(X,stock,names,start_year,end_year,ylabel):
    """
    Plots stock values from a portfolio X from start_year to end_year
    X : f8[n_stock,n_timestep]; Portfolio: open_value/close_value/variation of stocks
    stock : i4; stock under consideration from the portfolio X
    names : str[n_stocks]; Name of stocks in the portfolio
    start_year,end_year : f8; start and end year under consideration
    ylabel : str; ylabel of what is plotted ( open_value/close_value/variation)
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(start_year, end_year+1, 1)
    ax.set_xticks(major_ticks)
    ax.grid(which='both')
    plt.plot(np.linspace(start_year,end_year,len(X[:,stock])),X[:,stock])
    plt.xlabel('Time [years]'); plt.ylabel(ylabel + ' [USD]')
    plt.title(names[stock])
    plt.show()  
   
    
def k_means_ana(data,max_n_clusters,plot):
    """
    Analyses the best number of clusters according to the average silhouette score. 
    Silhouette coefficients (as these values are referred to as) near +1 indicate 
    that the sample is far away from the neighboring clusters. A value of 0 indicates 
    that the sample is on or very close to the decision boundary between two neighboring clusters
    and negative values indicate that those samples might have been assigned to the wrong cluster.
    IN:data[n_samples,m_characteristics] = data to analyse
       max_n_clusters = Integer - Maximum number of clusters
       plot=boolean  
    """
    #The range of clusters is 2-max_n_clusters
    range_n_clusters = np.arange(2,max_n_clusters+1)
    range_silhouette_avg = np.zeros(max_n_clusters-1)
    
    for n_clusters in range_n_clusters: #loop over different number of clusters
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        range_silhouette_avg[n_clusters-2] = silhouette_avg
        
    if plot == True:
        plt.figure()
        plt.plot(range_n_clusters,range_silhouette_avg,'*-')
        plt.title(f'Cluster Analysis for data with {len(data[1]):d} characteristics',fontsize=18)
        plt.xlabel('Number of clusters',fontsize=16)
        plt.ylabel('Average Silhouette Number',fontsize=16)
        
def k_means_trafo(data,symbols,n_clusters,plot):
    """
    Computes the centers of K-Means clusters
    If m_characteristics == 3 the clusters can be visualized
    IN: data[n_samples,m_characteristics] = data to analyse
        n_clusters = Integer - Number of clusters to find in the given dataset
    OUT: centers[n_centers,m_characteristics] = cluster centers for the given data  
         cluster_labels[n_samples] - cluster labels for data
    """
    #init: initialization method ('k-means++','random'); n_clusters:Integer - number of cluster centroids;
    #n_init: Integer - number of random walks executed to optimize results 
    clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    cluster_labels = clusterer.fit_predict(data)
    centers = clusterer.cluster_centers_
    
    if (plot == True and len(data[0]) <=3 ):
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(data, cluster_labels)
        
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1) #Plot: Silhouette Score
        fig.set_size_inches(13, 7)   #Format size of subplots
        archColors = cm.plasma(np.arange(n_clusters).astype(float) / (n_clusters)) #shape: (n_clusters,4)
        dataColors = cm.plasma(cluster_labels.astype(float) / n_clusters)          #shape: (n_samples,4)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this eg. the range is -0.2 to 1
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            #color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values, facecolors=archColors[i,:], alpha=1)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        # Labeling for subplot 1: Silhouette scores
        ax1.set_title(f"Silhouette plot for various clusters. Silhouette Average = {silhouette_avg:.3f}"
                       ,fontsize=18)
        ax1.set_xlabel("Silhouette coefficient values",fontsize=16)
        ax1.set_ylabel("Cluster label",fontsize=16)
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        # 2nd Plot showing the actual clusters formed
        if (len(data[0]) == 2): 
            ax2 = fig.add_subplot(1,2,2)
            #Plot data
            ax2.scatter(data[:,0], data[:,1], c=dataColors, alpha = 0.6, s=20)
            #Plot cluster centers
            #ax2.scatter(centers[:, 0], centers[:, 1],c=archColors,marker='X',edgecolor = 'k', alpha=1, s=200)
            ax2.set_title("K-Means clusteres",fontsize=18)
            ax2.set_xlabel('1st component',fontsize=16)
            ax2.set_ylabel('2nd component',fontsize=16)
            for i, txt in enumerate(symbols):
                ax2.annotate(txt, (data[i,0], data[i,1]))
            plt.tight_layout()
            plt.show()
            
        elif len(data[0] == 3):
            ax2 = fig.add_subplot(1,2,2, projection='3d')
            #Plot data
            ax2.scatter(data[:,0], data[:,1], data[:,2], c=dataColors, alpha = 0.6, s=5)
            #Plot cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],centers[:, 2],c=archColors,marker='X',edgecolor = 'k', alpha=1, s=200)
            ax2.set_title("K-Means clusteres of PCA reduced data.",fontsize=18)
            ax2.set_xlabel('1st pca component',fontsize=16)
            ax2.set_ylabel('2nd pca component',fontsize=16)
            ax2.set_zlabel('3rd pca component',fontsize=16)
            plt.tight_layout()
            plt.show()
    return centers,cluster_labels

def nodeplt(embedding,labels,symbols,partial_correlations,threshold):
    """
    Plots a map of the samples. The node positions is the 2D embedding of the cov of
    the samples. The colour indicates the group to which the sample belongs;
    The edges represent the connectivity to the other nodes (partial correlation)
    IN:    
    embedding : f8[2,n_samples]; A 2D embedding of the covariance matrix
    labels : labels[n_samples] - cluster labels for data 
    symbols : str[n_samples]; names of stocks in the portfolio
    partial_correlations : f8[n_samples,n_samples]; partial correlation between samples (connectivity)
    threshold : f8; value to cut off partial correlation
    """              
    x = embedding[0,:]; y = embedding[1,:]
    fig, ax = plt.subplots()
    ax.scatter(x, y,c=labels)
    for i, txt in enumerate(symbols):
        ax.annotate(txt, (x[i], y[i]))
    
    # Display a graph of the partial correlations
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > threshold) #tridiag entries where partial corr > 0.06 
    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label('Partial Correlation')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()
    
def find_cointegrated_pairs(data):
    n = data.shape[0]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    for i in range(n):
        print(i)
        for j in range(i+1, n):
            S1 = data[i]
            S2 = data[j]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((i, j))
    return score_matrix, pvalue_matrix, pairs

def zscore(X):
    return (X - X.mean()) / np.std(X)

def cointplt(data,pair,names,start_year,end_year,plot):
    index_s1 = pair[0]; index_s2 = pair[1]
    
    score, pvalue, _ = coint(data[index_s1],data[index_s2])
    ratio = data[index_s1] / data[index_s2]
    z_ratio = zscore(ratio) 
    if plot == True:
        X = z_ratio; 
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(start_year, end_year+1, 1)
        ax.set_xticks(major_ticks)
        ax.grid(which='both')
        
        plt.plot(np.linspace(start_year,end_year,len(X[:])),X[:])
        plt.axhline(z_ratio.mean(),color='black',ls='--')
        plt.axhline(1.0, color='red',ls='--')
        plt.axhline(-1.0, color='green',ls='--')
        
        plt.xlabel('Time [years]'); plt.ylabel('Ratio')
        plt.suptitle(names[pair[0]] + ' / ' + names[pair[1]])
        plt.title(f'Coint-Pvalue = {pvalue:.3} ')
        plt.show()  
    return pvalue

def hurstcoeff(close_val,pairs,names,plot):
    S1 = pairs[0]; S2 = pairs[1]
    ts = close_val[S1]/close_val[S2]
    lags = np.arange(2,400,1)
    tauvec = []
    for i in range(0,len(lags)):
        lag = lags[i]
        tau = np.sqrt(np.std(np.subtract(ts[lag:],ts[:-lag])))
        tauvec = np.append(tauvec,tau)
        
    poly = np.polyfit(np.log(lags), np.log(tauvec), 1) 
    H = poly[0]*2; #coeff orderd s.t. highest power first
    if plot == True:
        plt.figure()
        plt.scatter(np.log(lags),np.log(tauvec),color='black',s=2,label=r'$<|ln(S_{t+\tau})-ln(S_t)|^2>$')
        plt.plot(np.log(lags),poly[1]+np.log(lags)*0.5*H,color='green',
                 label=f'Linear fit: '+r'$f(\tau)$= '+f'{poly[1]:.2} + {poly[0]:.2}'+r'$\tau$' )
        plt.xlabel(r'$ln(\tau)$');
        plt.suptitle(f'Estimation of the Hurst coeff. H = {H:.2}')
        plt.title(names[pairs[0]] + '/' + names[pairs[1]])
        plt.legend(); plt.show()
    return H

def pairstrade(param,data,id0,id1,names,start_year,end_year,plot):
    S0 = data.iloc[id0][:]; S1 = data.iloc[id1][:]
    d,buy_signal,sell_signal = param[0],param[1],param[2] 
    d = int(d)
    k = (S0/S1).rolling(window=d, center=False).mean()
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
    buy,sell = np.asarray(buy),np.asarray(sell) 
    if plot == True and len(buy) != 0:
        size = (16,8)    
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(start_year, end_year+1, 1)
        ax.set_xticks(major_ticks)
        ax.grid(which='both')
        plt.plot(t,S0/S1, 'black',label='current k')
        plt.plot(t,k, 'green',label=f'{int(param[0])}-day moving average'); 
        plt.title('Stock Pair Ratio')
        plt.xlabel('t [years]'); plt.ylabel('k')
        plt.legend(); plt.show()
        #%%
        #Buy sell
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(start_year, end_year+1, 1);
        ax.set_xticks(major_ticks)
        ax.grid(which='both')
        plt.plot(t,S0.iloc[0:], 'black',label=names[id0]); plt.plot(t,S1.iloc[:], 'black', linestyle='dotted',label=names[id1]) 
        plt.scatter(buy[:,0],buy[:,1], color='g', linestyle='None', marker='^',label='Buy Signal')
        plt.scatter(sell[:,0],sell[:,1], color='r', linestyle='None', marker='^',label='Sell Signal')
        plt.title('Buy / Sell signal'); plt.xlabel('Time [years]'); plt.ylabel('Stock value [EUR]')
        plt.legend(); plt.show()
        
        #Money
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(start_year, end_year+1, 1)
        ax.set_xticks(major_ticks)
        ax.grid(which='both')
        plt.plot(t, money, 'black'); 
        plt.title('Trade performance');
        plt.xlabel('Time [years]'); plt.ylabel('Money [EUR]')
        plt.show()
         
    return money,buy,sell,k,Y_tilde,t,S0,S1