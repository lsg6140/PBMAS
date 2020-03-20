import numpy as np
from scipy.integrate import quad,dblquad

def breakage_discretize(Sfunc, bfunc, L, n, k):
    L = np.insert(L, 0, 0)
    bd = np.zeros((n, n))
    
    def num_func(x,y):
        return x**3 * Sfunc(y, k) * bfunc(x, y, k)
    
    def den_func(x):
        return x**3 * Sfunc(x, k)
    
    for i in range(n):
        den, err = quad(den_func, L[i], L[i+1])
        assert den != 0, 'breakage_discretize: division by zero'
        for j in range(i):
            num, err = dblquad(num_func, L[i], L[i+1],
                               lambda x: L[j], lambda x: L[j+1])
            Li = (L[i]+L[i+1])/2
            Lj = (L[j]+L[j+1])/2
            bd[j, i] = (Li / Lj)**3 * num / den
        num, err = dblquad(num_func, L[i], L[i+1],
                           lambda x: L[i], lambda x: x)
        bd[i, i] = num / den
        
    return bd 



def particle_number(bfunc, y, k): 
    Nb, err = quad(lambda x:bfunc(x, y, k), 0, y)
    return Nb



def selection_discretize(Sfunc, bfunc, L, n, k, bd):
    Sd = np.empty(n)
    L = np.insert(L, 0, 0)
    
    def integrand(y):
        int = (particle_number(bfunc, y, k) - 1) * Sfunc(y, k)
        return int
    
    for i in range(1, n):
        integ, err = quad(integrand, L[i], L[i+1])
        num = integ / (L[i+1] - L[i])
        sum = np.sum(bd[:i+1, i])
        den = sum - 1
        assert den != 0, 'selection_discretize: division by zero'
        Sd[i] = num / den
        
    Sd[0] = 0.0
    return Sd