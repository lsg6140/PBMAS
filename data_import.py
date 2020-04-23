import pandas as pd
import numpy as np

DTYPE = np.double

def importing(k0):
    df = pd.read_csv('particle_size.csv')
    data = df.to_numpy()
    length = data[:,0].astype(DTYPE)
    volume = data[:,1:].astype(DTYPE)
    volume /= 100 # Normalize to ratio
    number = np.empty(np.shape(volume), dtype=DTYPE) # Number density
    df2 = pd.read_csv('lognormal.csv')
    data2 = df2.to_numpy()
    mu = data2[0:4,-1][1:].astype(DTYPE)
    sigma = data2[4:,-1][1:].astype(DTYPE)
    time = np.array([0.,44.,88.,154.,330.,551.], dtype=DTYPE)

    n = np.size(length)
    N = np.size(time)
    p = np.size(k0)

    Q = np.diag(length**6).astype(DTYPE)

    # convert volume to number
    for i in range(N):
        number[:,i] = np.divide(volume[:,i], length**3)

    N0 = number[:,0]
    # Moments
    m0 = np.sum(N0)
    m1 = np.sum(length@N0)
    m2 = np.sum(np.power(length,2)@N0)
    m3 = np.sum(np.power(length,3)@N0)
    Y0 = np.append(N0,[m0,m1,m2,m3])
    
    return length, volume, number, Y0, mu, sigma, time, n, N, p, Q