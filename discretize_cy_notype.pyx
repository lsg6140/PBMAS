# cython: language_level=3

import numpy as np
from scipy.integrate import quad,dblquad
from lognormal_cy import selectionfunc, breakagefunc

def den_integrand(x, k, *args):
    return x**3 * selectionfunc(x, k, args)

def num_integrand(x, y, k, *args):
    return x**3 * selectionfunc(y, k, args) * breakagefunc(x, y, k, args)

def breakage_discretize(L, n, k, *args):
    L = np.insert(L, 0, 0)
    res = np.zeros((n, n))

    for i in range(n):
        den, err = quad(den_integrand, L[i], L[i+1], args=(k, *args))
        assert den != 0, 'breakage_discretize: division by zero'
        for j in range(i):
            num, err = dblquad(num_integrand, L[i], L[i+1],
                               lambda x: L[j], lambda x: L[j+1],
                               args=(k, *args))
            Li = (L[i]+L[i+1])/2
            Lj = (L[j]+L[j+1])/2
            res[j, i] = (Li / Lj)**3 * num / den
        num, err = dblquad(num_integrand, L[i], L[i+1],
                           lambda x: L[i], lambda x: x,
                           args=(k, *args))
        res[i, i] = num / den
        
    return res 

def particle_number(x, k, *args): 
    res = quad(lambda a: breakagefunc(a, x, k, args), 0, x)[0]
    return res

def selection_integrand(x, k, *args):
    return (particle_number(x, k, *args) - 1) * selectionfunc(x, k, args)

def selection_discretize(L, n, k, breakage_mat, *args):
    res = np.empty(n)
    L = np.insert(L, 0, 0)
    
    for i in range(1, n):
        integ = quad(selection_integrand, L[i], L[i+1], args=(k, *args))[0]
        num = integ / (L[i+1] - L[i])
        sum = np.sum(breakage_mat[:i+1, i])
        den = sum - 1
        assert den != 0, 'selection_discretize: division by zero'
        res[i] = num / den
        
    res[0] = 0.0
    return res

if __name__ == '__main__':
    n = 10
    k = [1.0, 2.0]
    p = len(k)
    L = np.linspace(1.0, 20, n)
    breakge_matrix = breakage_discretize(L, n, k)
    selection_vector = selection_discretize(L, n, k, breakge_matrix)