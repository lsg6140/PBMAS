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

        
        
##########################################################################################        




def discretize_cython_check():
    import discretize       # this imports lognormal_cy.py
    import discretize_cy
    import discretize_py    # this imports lognormal.py
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
    
    
    args = [mu, sigma]
    
    breakage = discretize_py.breakage_discretize(length, n, k0, *args)
    selection = discretize_py.selection_discretize(length, n, k0, breakage, *args)
    
    python_break = discretize_py.breakage_discretize(length, n, k0, *args)
    python_selec = discretize_py.selection_discretize(length, n, k0, python_break, *args)
    
    cython_break = discretize_cy.breakage_discretize(length, n, k0, *args)
    cython_selec = discretize_cy.selection_discretize(length, n, k0, cython_break, *args)
    
    assert np.allclose(breakage, python_break, cython_break), 'breakages not match'
    assert np.allclose(selection, python_selec, cython_selec), 'selections not match'
    print('No error')   
    
        
def discretize_parallel_check():
    import discretize
    import discretize_parallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
    
    
    args = [mu, sigma]
    
    serial_break = discretize.breakage_discretize(length, n, k0, *args)
    serial_selec = discretize.selection_discretize(length, n, k0, serial_break, *args)
    
    parallel_break = discretize_parallel.breakage_discretize(length, n, k0, *args)
    parallel_selec = discretize_parallel.selection_discretize(length, n, k0, parallel_break, *args)

    assert np.allclose(serial_break, parallel_break), 'breakages not match'
    assert np.allclose(serial_selec, parallel_selec), 'selections not match'
    print('No error')    
    
    
def discretize(method='python', it=1):
    import discretize
    import discretize_py
    import discretize_cy
    import discretize_parallel
    
    k0 = np.array([1e-1,0.8,0.15], dtype=DTYPE)
    length, _, _, _, mu, sigma, _, n, _, _, _ = importing(k0)
       
    args = [mu, sigma]    
    
    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            bmat = discretize_py.breakage_discretize(length, n, k0, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of breakage takes %6.3f s.' % aver_time)
        
        total = 0
        for i in range(it):
            tic = time.time()
            res = discretize_py.selection_discretize(length, n, k0, bmat, *args)
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
        
    elif method == 'serial':
        it *= 10
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
            

    elif method == 'parallel':
        it *= 10
        total = 0
        for i in range(it):
            tic = time.time()
            bmat = discretize_parallel.breakage_discretize(length, n, k0, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of breakage takes %6.3f s.' % aver_time)
        
        total = 0
        for i in range(it):
            tic = time.time()
            res = discretize_parallel.selection_discretize(length, n, k0, bmat, *args)
            toc = time.time()
            total += toc - tic
        aver_time = total / it
        print('discretization of selection takes %6.3f s.' % aver_time)
        
        
########################################################################################## 



        
def ode_check():
    import discretize
    import ode
    import ode_cy
    import ode_cy_parallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    number0 = number[:,0]
    
    
    args = [mu, sigma]
    
    brk_mat, slc_vec = discretize.discretize(length, n, p, k0, 1e-8, *args)[0:2]
    
    dndt_py = ode.breakage(number0, brk_mat, slc_vec)
    dndt_cy = ode_cy.breakage(number0, brk_mat, slc_vec)
    dndt_parallel = ode_cy_parallel.breakage(number0, brk_mat, slc_vec)
    
    assert np.allclose(dndt_py, dndt_cy, dndt_parallel), 'Error in ode'
    print('No error')
    
    
def ode(method='python', it=100000):
    import discretize
    import ode
    import ode_cy
    import ode_cy_parallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    number0 = number[:,0]
    
    
    args = [mu, sigma]
    
    brk_mat, slc_vec = discretize.discretize(length, n, p, k0, 1e-8, *args)[0:2]
    
    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            res = ode.breakage(number0, brk_mat, slc_vec)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('constructing ode takes %5.2f \u03BCs.' % aver_time)
            

    elif method == 'cython':
        total = 0
        it *= 50
        for i in range(it):
            tic = time.time()
            res = ode_cy.breakage(number0, brk_mat, slc_vec)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('constructing ode takes %5.2f \u03BCs.' % aver_time)
        
    elif method == 'parallel':
        total = 0
        for i in range(it):
            tic = time.time()
            res = ode_cy_parallel.breakage(number0, brk_mat, slc_vec)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e6
        print('constructing ode takes %5.2f \u03BCs.' % aver_time)
        
        
##########################################################################################     
    
        
    
    
def phi_check():
    import discretize
    import phi
    import phi_cy
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    number0 = number[:,0]
    
    
    args = [mu, sigma]
    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = number0.copy()
    
        
    dbs = discretize.discretize(length, n, p, k0, 1e-8, *args)
    
    dzdt_py = phi.phi_breakage(Z0, dbs, n, p, 1e-8)
    dzdt_cy = phi_cy.phi_breakage(Z0, dbs, n, p, 1e-8)
    
    assert np.allclose(dzdt_py, dzdt_cy), 'Error in phi'
    print('No error')
    
    
def phi_parallel_check():
    import discretize
    import phi_cy
    import phi_cy_parallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    number0 = number[:,0]
    
    
    args = [mu, sigma]
    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = number0.copy()
    
        
    dbs = discretize.discretize(length, n, p, k0, 1e-8, *args)
    
    dzdt_serial = phi_cy.phi_breakage(Z0, dbs, n, p, 1e-8)
    dzdt_parallel = phi_cy_parallel.phi_breakage(Z0, dbs, n, p, 1e-8)
    
    assert np.allclose(dzdt_serial, dzdt_parallel), 'Error in phi'
    print('No error')
    
    
    
def phi(method='python', it=1000):
    import discretize
    import phi
    import phi_cy
    import phi_cy_parallel
    
    k0 = np.array([1e-7,0.8,0.15], dtype=DTYPE)
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    number0 = number[:,0]
    
    
    args = [mu, sigma]
    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = number0.copy()
    
    dbs = discretize.discretize(length, n, p, k0, 1e-8, *args)
    
    if method == 'python':
        total = 0
        for i in range(it):
            tic = time.time()
            res = phi.phi_breakage(Z0, dbs, n, p, 1e-8)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e3
        print('constructing phi takes %5.2f ms.' % aver_time)
            
    elif method == 'cython':
        it *= 10
        total = 0
        for i in range(it):
            tic = time.time()
            res = phi_cy.phi_breakage(Z0, dbs, n, p, 1e-8)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e3
        print('constructing phi takes %5.2f ms.' % aver_time)
        
    elif method == 'parallel':
        it *= 10
        total = 0
        for i in range(it):
            tic = time.time()
            res = phi_cy_parallel.phi_breakage(Z0, dbs, n, p, 1e-8)
            toc = time.time()
            total += toc - tic
        aver_time = total / it * 1e3
        print('constructing phi takes %5.2f ms.' % aver_time)