import numpy as np
from scipy.integrate import quad,dblquad
from lognormal_cy import selectionfunc, breakagefunc
from joblib import Memory, Parallel, delayed

cachedir = './cachedir'
memory = Memory(cachedir, verbose=0)

def den_integrand(x, k, *args):
    return x**3 * selectionfunc(x, k, args)

def num_integrand(x, y, k, *args):
    return x**3 * selectionfunc(y, k, args) * breakagefunc(x, y, k, args)

def breakage_discretize(L, n, k, *args):
    L = np.insert(L, 0, 0)
    
    def in_for_loop(i):
        temp = np.zeros(n)
        den, err = quad(den_integrand, L[i], L[i+1], args=(k, *args))
        assert den != 0, 'breakage_discretize: division by zero'
        for j in range(i):
            num, err = dblquad(num_integrand, L[i], L[i+1],
                               lambda x: L[j], lambda x: L[j+1],
                               args=(k, *args))
            Li = (L[i]+L[i+1])/2
            Lj = (L[j]+L[j+1])/2
            temp[j] = (Li / Lj)**3 * num / den
        num, err = dblquad(num_integrand, L[i], L[i+1],
                           lambda x: L[i], lambda x: x,
                           args=(k, *args))
        temp[i] = num / den
        
        return temp
    
    r = Parallel(n_jobs=-1)(delayed(in_for_loop)(i) for i in range(n))
    
    res = np.stack(r).T 
        
    return res 



def particle_number(x, k, *args): 
    res = quad(lambda a: breakagefunc(a, x, k, args), 0, x)[0]
    return res

def selection_integrand(x, k, *args):
    return (particle_number(x, k, *args) - 1) * selectionfunc(x, k, args)

def selection_discretize(L, n, k, breakage_mat, *args):
    L = np.insert(L, 0, 0)
    
    def in_for_loop(i):
        integ = quad(selection_integrand, L[i], L[i+1], args=(k, *args))[0]
        num = integ / (L[i+1] - L[i])
        sum = np.sum(breakage_mat[:i+1, i])
        den = sum - 1
        assert den != 0, 'selection_discretize: division by zero'
        return num / den
        
    r = Parallel(n_jobs=-1)(delayed(in_for_loop)(i) for i in range(1, n))
    
    res = np.zeros(n)
    res[1:] = r
    return res

@memory.cache
def discretize(L, n, p, k, delta, *args):
    print('discretizing...')
    brk_mat = np.empty((n, n))
    slc_vec = np.empty(n)
    brk_mat_r = np.empty((p, n, n))
    brk_mat_l = np.empty((p, n, n))
    slc_vec_r = np.empty((p, n))
    slc_vec_l = np.empty((p, n))
    
    K = np.tile(k, [p, 1])    
    Kr = K + np.eye(p) * delta
    Kl = K - np.eye(p) * delta
    
    brk_mat = breakage_discretize(L, n, k, *args)
    slc_vec = selection_discretize(L, n, k, brk_mat, *args)
    
    for i in range(p):
        brk_mat_r[i] = breakage_discretize(L, n, Kr[i], *args)
        brk_mat_l[i] = breakage_discretize(L, n, Kl[i], *args)
        slc_vec_r[i] = selection_discretize(L, n, Kr[i], brk_mat_r[i], *args)
        slc_vec_l[i] = selection_discretize(L, n, Kl[i], brk_mat_l[i], *args)
        
    return brk_mat, slc_vec, brk_mat_r, slc_vec_r, brk_mat_l, slc_vec_l

if __name__ == '__main__':
    n = 10
    k = [1.0, 2.0]
    p = len(k)
    L = np.linspace(1.0, 20, n)
    breakge_matrix = breakage_discretize(L, n, k)
    selection_vector = selection_discretize(L, n, k, breakge_matrix)