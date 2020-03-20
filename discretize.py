import numpy as np
from scipy.integrate import quad,dblquad

def num_integrand(x, y, k, *args):
    return x**3 * selection_szdp(y, k) * break_lognormal(x, y, k, args[0], args[1])

def den_integrand(x, k, *args):
    return x**3 * selection_szdp(x, k)

def breakage_discretize(L, n, k, *args):
    L = np.insert(L, 0, 0)
    bd = np.zeros((n, n))

    for i in range(n):
        den, err = quad(den_integrand, L[i], L[i+1], args=(k, *args))
        assert den != 0, 'breakage_discretize: division by zero'
        for j in range(i):
            num, err = dblquad(num_integrand, L[i], L[i+1],
                               lambda x: L[j], lambda x: L[j+1],
                               args=(k, *args))
            Li = (L[i]+L[i+1])/2
            Lj = (L[j]+L[j+1])/2
            bd[j, i] = (Li / Lj)**3 * num / den
        num, err = dblquad(num_integrand, L[i], L[i+1],
                           lambda x: L[i], lambda x: x,
                           args=(k, *args))
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