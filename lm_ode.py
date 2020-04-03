import numpy as np
from numpy import linalg as LA

from solve_tools import *

def lm_ode(phi, yhat, t, params, Q, opts=[1e-3, 1e-8, 1e-8, 1000], *args):
    # Input arguments

    # opts = [tau, tolg, tolk, max_iter]
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
        fail = False
        nu = 2
        it = 0 
        rho = 0
        ter = 'm'
        delta = 1e-8
        N = np.size(t)
        
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
        
        f, Y, hessian, gradient, r, fail = get_fYHg(phi, yhat, t, k, n, p, N, Q, scalar, delta, *args)
       
        assert not fail, "solving ODE with guessed parameters failed"
        f_old = f
        r_old = r.copy()
        gn = LA.norm(gradient, np.inf)
        stop = bool(gn < opts[1])
        
        if stop:
            ter = 'gradient'
            print("First guess was correct!")
            
        mu = opts[0] * max(np.diag(hessian))
        print('Iter | Obj func | step size | gradient |   mu   |   rho')
        print("{0:5d}|{1:10.4e}|   Not cal |  Not cal |{2:8.1e}| Not cal"
              .format(it, f, mu))
        
        while (not stop) and (it <= opts[3]):
            it += 1
            h, mu = cholsolve(hessian, -gradient, mu, p)
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if hn <= opts[2] * (kn + opts[2]):
                print('termination by stepsize')
                stop = True
                ter = 'h'
            else:
                k_new = k + h
                
                f, Y, hessian_new, gradient_new, r, fail = get_fYHg(phi, yhat, t, k_new, n, p, N, Q, scalar, delta, *args)
                if fail:
                    f = f_old
                else:
                    hessian = hessian_new
                    gradient = gradient_new
                if f > f_old:
                    fail = True

                dL = h @ (mu * h - gradient) / 2
                
                if dL > 0 and not fail:
                    df = dF(r_old, r, Q, n, N)
                    rho = df / dL
                    k = k_new 
                    f_old = f
                    r_old = r.copy()
                    gn = LA.norm(gradient, np.inf) 
                    
                    if gn < opts[1]:
                        print('termination by gradient')
                        stop = True
                        ter = 'gradient'
                    mu *= max(1/3,1-(2*rho-1)**3)
                    nu = 2
                else:
                    mu *= nu
                    nu *= 2

            if not stop:
                if rho == 0:
                    print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}| Not cal"
                        .format(it, f, hn, gn, mu))
                else:
                    print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}|{5:8.1e}"
                            .format(it, f, hn, gn, mu, rho))
                
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
    
    k0 = np.array([8e-7, 0.5, 0.4])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    res = lm_ode(pbm.phi, number, t, k0, Q, [1e-3, 1e-8, 1e-8, 1000], *args)