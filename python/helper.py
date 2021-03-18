import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sklearn import preprocessing
from sklearn import metrics

def plot_points(X,y,center=None,title='',output1=None):
    
    pepcolor=np.unique(y)
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10))
    fig.subplots_adjust(top=0.85)
    fig.suptitle('\n'+title,fontsize=24)
    listcolor=['plum',
              'darkgreen',
              'gold',
               'red',
               'yellowgreen',
               'blue',
              'black',
              'lightcoral',
              'cornflowerblue',
              'darkgrey']
    
    # marker shape
    for n, i in enumerate(pepcolor):
        mask=np.where(y==i)
        Xx=X[mask,0]
        Xy=X[mask,1]
        ax.scatter(Xx,Xy,
                 color=listcolor[n],
                  #s=2,
                 label=i)
        if center is not None:
            ax.scatter(center[n,0],center[n,1],
                       marker='^',
#                        color=listcolor[n],
                       s=100,
                       facecolors='white',
                   edgecolors=listcolor[n]
                      )
            
#     ax.set_ylabel(r'$-\log_{10} (P)$',fontsize=18)
#     ax.set_xlabel('$\log_2$(fold change)',fontsize=18)
#     ax.set_ylim(-0.5,51)
    ax.tick_params(labelsize=14)
#    if you use fig.legend, legend will be outside figure
    plt.legend(#loc='upper left',
    prop={'size': 14}
    )
    if output1 is not None:
        fig.savefig(output1)

def plot_bar(X,y,cmapper,title='',output1=None):
    """
    X: 2-d array of xy-coordinates
    y: 1-d target array of numbers
    cmapper: labelencoder we defined to convert category into numbers
    """

       
    pepcolor=np.unique(y)
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(14,10))
    fig.subplots_adjust(top=0.85)
    plt.suptitle('\n'+title,fontsize=24) 
    
    plt.scatter(*X.T, c=y, 
        cmap='Set1', 
        alpha=1.0
        )
            

    ax.tick_params(labelsize=14)

    # plt.setp(ax, xticks=[], yticks=[])
    # you need a mapper object to plot cbar
    cbar = plt.colorbar(boundaries=np.arange(len(pepcolor)+1)-0.5)
    cbar.set_ticks(np.arange(len(pepcolor)))
    cbar.set_ticklabels(cmapper.inverse_transform(np.arange(len(pepcolor))))
    cbar.ax.tick_params(labelsize=14)
    
    if output1 is not None:
        fig.savefig(output1)
    plt.show()
def plot_cmatrix(ytrue,ypred,cmapper,title='',output1=None):

    c1=metrics.confusion_matrix(ytrue,ypred)
    pepcolor=np.unique(ytrue)
    fig, ax = plt.subplots()
    # fig.subplots_adjust(top=0.85)
    plt.suptitle('\n'+title) 
    
    ax=sns.heatmap(c1/np.sum(c1,axis=0),fmt='.1%',
        annot=True,cmap='Blues',
        xticklabels=cmapper.inverse_transform(np.arange(len(pepcolor))),
        yticklabels=cmapper.inverse_transform(np.arange(len(pepcolor)))
    )
    ax.tick_params(labelsize=14)
    if output1 is not None:
        fig.savefig(output1)
    plt.show()
