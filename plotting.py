import matplotlib.pyplot as plt
import numpy as np

def plotvolume(Y, volume, time, length):
    V = np.empty(np.shape(Y))
    for i in range(6):
        V[:,i] = np.multiply(Y[:,i], length**3)
    fig, axes = plt.subplots(6,1,figsize=(5,10))
    for i in range(6):
        axes[i].plot(length,V[:,i])
        axes[i].scatter(length,volume[:,i])
        axes[i].set_xscale('log')
        axes[i].set_xlim([length[0],length[-1]])
        axes[i].title.set_text('t={0}'.format(time[i]))
    fig.show()
    
def plotnumber(Y, yhat, time, length):
    fig, axes = plt.subplots(6,1,figsize=(5,10))
    for i in range(6):
        axes[i].plot(length,Y[:,i])
        axes[i].scatter(length,yhat[:,i])
        axes[i].set_xscale('log')
        axes[i].set_xlim([length[0],length[-1]])
        axes[i].title.set_text('t={0}'.format(time[i]))
    fig.show()