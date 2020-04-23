import numpy as np
from numpy import linalg as LA

from solve_tools import *

def gn_ode(phi, yhat, t, params, Q, opts=[1e-8, 1e-8, 1000], *args):
    # Input arguments

    # opts = [tolg, tolk, max_iter]
    #
    # Outputs
    # output = [k,Y,info]
    # k : parameters
    # Y : results with k
    # info = [it,ter]
    # it : Number of iterations
    # ter : Termination criteria 1: gradient 2: change in h 3: maximum iteration

    try:
        stop = False
        it = 0 
        ter = 'm'
        delta = 1e-8
        N = np.size(t)
        
        kmax = 100
        
        if np.ndim(yhat) == 1:
            n = 1
            assert N == np.size(yhat), "gn_ode: dimension mismatch with yhat and t"
        else:
            n = np.size(yhat, 0)
            assert N == np.size(yhat, 1), "gn_ode: dimension mismatch with yhat and t"
            
        p = np.size(params)
        k = params.copy()
        
        f, Y, hessian, gradient, r, fail =  get_fYHg(phi, yhat, t, k, n, p, N, Q, delta, *args)
        
        assert not fail, 'gn_ode: Solving ODE with first guess failed. Try with another parameters.'
         
        gn = LA.norm(gradient, np.inf)
        stop = bool(gn < opts[0])
        
        if stop:
            ter = 'gradient'
            print("gn_ode: First guess was correct!")
        
        while (not stop) and (it <= opts[2]):
            fail = False
            it += 1
            
            K = np.diag(k)
            gr = gradient @ K
            Hr = K @ hessian @ K
            
            hr, _ = cholsolve(Hr, -gr, 0, p)
            h = K @ hr
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if it == 1:
                print('Iter | Obj func | step size | gradient ')
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}"
                      .format(it, f, hn, gn))
            else:
                print('Iter | Obj func | step ratio | gradient ')
                print("{0:5d}|{1:10.4e}|{2:12.3e}|{3:10.2e}"
                      .format(it, f, hn/hn_old, gn))
            
            if hn <= opts[1] * (kn + opts[1]):
                print('gn_ode: termination by stepsize')
                stop = True
                ter = 'h'
            else:
                print('gn_ode: current step is', h)
                f, k, Y, hessian, gradient, func_eval = soft_linesearch(f, h, gradient, phi, yhat, t, k, n, p, N, Q, delta, kmax, *args)
                
                gn = LA.norm(gradient, np.inf)
                if gn < opts[0]:
                    print('gn_ode: termination by gradient')
                    stop = True
                    ter = 'gradient'

            hn_old = hn
        
        if ter == 'gradient':
            print('Iter | Obj func | step size | gradient ')
            print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}"
                  .format(it, f, hn, gn))
            
        info = [it, ter]
        output = [k, Y, info]
        return output
    
    except OverflowError as oerr:
        print(oerr.args)
        return
    
    except AssertionError as aerr:
        print(aerr.args)
        return   

def gn_ode_bisection(phi, yhat, t, params, Q, c=True, opts=[1e-8, 1e-8, 1000], *args):
    # Input arguments

    # opts = [tolg, tolk, max_iter]
    #
    # Outputs
    # output = [k,Y,info]
    # k : parameters
    # Y : results with k
    # info = [it,ter]
    # it : Number of iterations
    # ter : Termination criteria 1: gradient 2: change in h 3: maximum iteration

    try:
        stop = False
        it = 0 
        ter = 'm'
        delta = 1e-8
        N = np.size(t)
        
        if np.ndim(yhat) == 1:
            n = 1
            assert N == np.size(yhat), "gn_ode: dimension mismatch with yhat and t"
        else:
            n = np.size(yhat, 0)
            assert N == np.size(yhat, 1), "gn_ode: dimension mismatch with yhat and t"
            
        p = np.size(params)
        k = params.copy()
        
        f, Y, hessian, gradient, r, fail =  get_fYHg(phi, yhat, t, k, n, p, N, Q, delta, *args)
        
        assert not fail, 'gn_ode: Solving ODE with first guess failed. Try with another parameters.'
         
        gn = LA.norm(gradient, np.inf)
        stop = bool(gn < opts[0])
        
        if stop:
            ter = 'gradient'
            print("gn_ode: First guess was correct!")
        
        final_print = True
        while (not stop) and (it <= opts[2]):
            fail = False
            
            K = np.diag(k)
            gr = gradient @ K
            Hr = K @ hessian @ K
            
            hr, _ = cholsolve(Hr, -gr, 0, p)
            h = K @ hr
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if it == 0:
                print('Iter | Obj func | step size | gradient ')
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}"
                      .format(it, f, hn, gn))
            else:
                print('Iter | Obj func | step size | gradient ')
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}"
                      .format(it, f, hn * mu, gn))
            
            if hn <= opts[1] * (kn + opts[1]):
                print('gn_ode: termination by stepsize')
                stop = True
                final_print = False
                ter = 'h'
            else:
                it += 1
                print('gn_ode: current step is', h)
                f, k, Y, hessian, gradient, mu, func_eval = bisection(f, h, gradient, phi, yhat, t, k, n, p, N, Q, delta, c, *args)
                
                gn = LA.norm(gradient, np.inf)
                if gn < opts[0]:
                    print('gn_ode: termination by gradient')
                    stop = True
                    ter = 'gradient'
                if hn * mu <= opts[1] * (kn + opts[1]):
                    print('gn_ode: termination by stepsize')
                    stop = True
                    ter = 'h'
        
        if final_print:
            print('Iter | Obj func | step size | gradient ')
            print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}"
                  .format(it, f, hn * mu, gn))
            
        info = [it, ter]
        output = [k, Y, info]
        return output
    
    except OverflowError as oerr:
        print(oerr.args)
        return
    
    except AssertionError as aerr:
        print(aerr.args)
        return   
    

    

if __name__ == '__main__':
    
    import pbm
    
    from data_import import importing
    
    k0 = np.array([1e-7, 0.8, 0.15])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    res = gn_ode(pbm.phi, number, t, k0, Q, [1e-8, 1e-8, 1000], *args)