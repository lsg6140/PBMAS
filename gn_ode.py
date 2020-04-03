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
        nu = 2
        it = 0 
        rho = 0
        ter = 'm'
        delta = 1e-8
        N = np.size(t)
        
        amax = 10
        kmax = 100
        
        if np.ndim(yhat) == 1:
            scalar = True
            n = 1
            assert N == np.size(yhat), "Dimension mismatch with yhat and t"
        else:
            scalar = False
            n = np.size(yhat, 0)
            assert N == np.size(yhat, 1), "Dimension mismatch with yhat and t"
            
        p = np.size(params)
        k = params.copy()
        
        f, Y, hessian, gradient, r, fail =  get_fYHg(phi, yhat, t, k, n, p, N, Q, scalar, delta, *args)
        
        assert not fail, 'Solving ODE with first guess failed. Try with another parameters.'
         
        gn = LA.norm(gradient, np.inf)
        stop = bool(gn < opts[0])
        
        if stop:
            ter = 'gradient'
            print("First guess was correct!")
            
        print('Iter | Obj func | step size | gradient ')
        print("{0:5d}|{1:10.4e}|   Not cal |{2:10.2e}".format(it, f, gn))
        
        while (not stop) and (it <= opts[2]):
            fail = False
            it += 1
            h, _ = cholsolve(hessian, -gradient, 0, p)
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if hn <= opts[1] * (kn + opts[1]):
                print('termination by stepsize')
                stop = True
                ter = 'h'
            else:
                f, k, Y, hessian, gradient = soft_linesearch(f, h, gradient, phi, yhat, t, k, n, p, N, Q, scalar, delta, amax, kmax, *args)
                
                gn = LA.norm(gradient, np.inf)
                if gn < opts[0]:
                    print('termination by gradient')
                    stop = True
                    ter = 'gradient'

            if not stop:
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
    

    

if __name__ == '__main__':
    
    import pbm
    
    from data_import import importing
    
    k0 = np.array([1e-7, 0.8, 0.15])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    res = gn_ode(pbm.phi, number, t, k0, Q, [1e-8, 1e-8, 1000], *args)