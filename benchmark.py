import numpy as np
import time
from data_import import importing

DTYPE = np.double

def breakage(method='python', it=100000):
    import lognormal
    import lognormal_cy
    
    mu = np.array([1.0, 2.0, 3.0], dtype=DTYPE)
    sigma = np.array([2.0, 2.0, 2.0], dtype=DTYPE)
    x = 0.1
    y = 3.0
    k = np.array([0.1, 0.2, 0.3], dtype=DTYPE)
    args = [mu, sigma]

    python_res = lognormal.breakagefunc(x, y, k, *args)

    cython_res = lognormal_cy.breakagefunc(x, y, k, args)
    
    assert python_res == cython_res, 'Values not match'

    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            res = lognormal.breakagefunc(x, y, k, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('breakage function takes %5.2f \u03BCs.' % aver_time)
            

    elif method == 'cython':
        total = 0
        for i in range(it):
            tic = time.time()
            res = lognormal_cy.breakagefunc(x, y, k, args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('breakage function takes %5.2f \u03BCs.' % aver_time)
        
        
def selection(method='python', it=10000000):
    import lognormal
    import lognormal_cy

    mu = np.array([1.0, 2.0, 3.0], dtype=DTYPE)
    sigma = np.array([2.0, 2.0, 2.0], dtype=DTYPE)
    x = 0.1
    y = 3.0
    k = np.array([0.1, 0.2, 0.3], dtype=DTYPE)
    args = [mu, sigma]

    python_res = lognormal.selectionfunc(x, k, *args)
    
    cython_res = lognormal_cy.selectionfunc(x, k, args)
    
    assert python_res == cython_res, 'Values not match'
    
    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            res = lognormal.selectionfunc(x, k, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('breakage function takes %5.2f \u03BCs.' % aver_time)
            

    elif method == 'cython':
        total = 0
        for i in range(it):
            tic = time.time()
            res = lognormal_cy.selectionfunc(x, k, args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('breakage function takes %5.2f \u03BCs.' % aver_time)
        

        
def parallel_check():
    import discretize
    import discretize_noparallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
    
    
    args = [mu, sigma]
    
    python_break = discretize.breakage_discretize(length, n, k0, *args)
    python_selec = discretize.selection_discretize(length, n, k0, python_break, *args)
    
    noparallel_break = discretize_noparallel.breakage_discretize(length, n, k0, *args)
    noparallel_selec = discretize_noparallel.selection_discretize(length, n, k0, noparallel_break, *args)

    assert np.allclose(python_break, noparallel_break), 'breakages not match'
    assert np.allclose(python_selec, noparallel_selec), 'selections not match'
    print('No error')

def discretize_check():
    import discretize
    import discretize_cy
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
    
    
    args = [mu, sigma]
    
    python_break = discretize.breakage_discretize(length, n, k0, *args)
    python_selec = discretize.selection_discretize(length, n, k0, python_break, *args)
    
    cython_break = discretize_cy.breakage_discretize(length, n, k0, *args)
    cython_selec = discretize_cy.selection_discretize(length, n, k0, cython_break, *args)
    
    assert np.allclose(python_break, cython_break), 'breakages not match'
    assert np.allclose(python_selec, cython_selec), 'selections not match'
    print('No error')   
    
def discretize(method='python', it=1):
    import discretize
    import discretize_cy
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
       
    args = [mu, sigma]    
    
    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            bmat = discretize.breakage_discretize(length, n, k0, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of breakage takes %6.3f s.' % aver_time)
        
        total = 0
        for i in range(it):
            tic = time.time()
            res = discretize.selection_discretize(length, n, k0, bmat, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of selection takes %6.3f s.' % aver_time)
            

    elif method == 'cython':
        total = 0
        for i in range(it):
            tic = time.time()
            bmat = discretize_cy.breakage_discretize(length, n, k0, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of breakage takes %6.3f s.' % aver_time)
        
        total = 0
        for i in range(it):
            tic = time.time()
            res = discretize_cy.selection_discretize(length, n, k0, bmat, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of selection takes %6.3f s.' % aver_time)
        
    