import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import time

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
    

def plot(params=np.array([1e-1, 0.8, 0.15])):
    from data_import import importing
    from solve_ode import solve_jac
    import pbm
    
    
    k0 = np.asarray(params)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    tic = time.time()
    Y, Jac = solve_jac(pbm.phi, number, t, k0, n, p, N, False, 1e-8, *args)
    toc = time.time() - tic
    print('solving ODE took %5.2f seconds' % toc)

    plotvolume(Y, volume, t, length)
    

if __name__ == '__main__':
    plot()