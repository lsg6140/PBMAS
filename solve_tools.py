import numpy as np
from numpy import linalg as LA
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize

from solve_ode import evolve, evolve_onestep

def get_fYHg(phi, yhat, t, params, n, p, N, Q, delta, *args):
    # initial condition J0 = 0
    if n == 1:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
        
    r = np.zeros((n, N))    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()
    f = 0
    fail = False    
    
    print('get_fYHg: solve with parameters', params)
    Z, suc = evolve(phi, Z0, t, params, n, p, delta, *args)
    
    if not suc:
        print('get_fYHg: solving ODE failed')
        fail = True
        return f, yhat, np.zeros((p, p)), np.zeros(p), r, fail
            
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    r = yhat - Y
    
    for i in range(N):
        f += r[:, i] @ Q @ r[:, i] / 2
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    H = np.zeros((p, p))
    g = np.zeros(p)
    
    for i in range(N):
        JQ = Jt[i].T @ Q
        H += JQ @ Jt[i]
        g -= JQ @ r[:, i]
        
    return f, Y, H, g, r, fail        


def get_fYHg_stepwise(f_old, phi, yhat, t, params, n, p, N, Q, delta, *args):
    # initial condition J0 = 0
    if n == 1:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
        
    r = np.zeros((n, N))    
    Z = np.zeros((n * (p + 1), N))
    Z[0:n, 0] = y0.copy()
    f = 0
    fail = False    
    
    print('get_fYHg_stepwise: solve with parameters', params)
    for i in range(N-1):        
        Z[:, i+1], suc = evolve_onestep(phi, Z[:, i], [t[i], t[i+1]], params, n, p, delta, *args)
        if not suc:
            print('get_fYHg_stepwise: solving ODE failed')
            fail = True
            return f, yhat, np.zeros((p, p)), np.zeros(p), r, fail
        r[:, i+1] = yhat[:, i+1] - Z[0:n, i+1]
        f += r[:, i+1] @ Q @ r[:, i+1] / 2
        if f > f_old:
            print('get_fYHg_stepwise: Error is bigger than previous step')
            fail = True
            return f, yhat, np.zeros((p, p)), np.zeros(p), r, fail
            
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    H = np.zeros((p, p))
    g = np.zeros(p)
    
    for i in range(N):
        JQ = Jt[i].T @ Q
        H += JQ @ Jt[i]
        g -= JQ @ r[:, i]
        
    return f, Y, H, g, r, fail 


def dF(r_old, r , Q, n, N):
    df = 0
    
    if n == 1:
        df = (r_old[0] - r[0]) @ (r_old[0] + r[0]) / 2
    else:
        for i in range(N):
            df += (r_old[:,i] - r[:,i]) @ Q @ (r_old[:,i] + r[:,i]) /2
            
    return df
   
    
    
def cholesky(A, p):
    C = np.zeros((p,p))
    j = 0
    pd = True
    
    while pd and j < p:
        sum = 0
        
        for k in range(j):
            sum += C[j, k]**2
            
        d = A[j, j] - sum
        
        if d > 0:
            C[j, j] = np.sqrt(d)
            
            for i in range(j,p):
                sum = 0
                for k in range(j):
                    sum += C[i, k] * C[j, k]
                C[i, j] = (A[i, j]-sum) / C[j, j]
                
        else:
            pd = False
            
        j += 1
        
    return C, pd



def cholsolve(A, b, mu, p):
    I = np.eye(p)
    mA = np.amax(abs(A))
    
    
    if mu != 0:
        pd = False

        while pd == False:
            C, pd = cholesky(A + mu * I, p)

            # check for near singularity
            if pd == True:
                pd = (1 / LA.cond(C,1) >= 1e-15)
            if pd == False:
                mu = max(10 * mu, np.finfo(float).eps * mA)
    else:
        C, pd = cholesky(A, p)
        assert pd, "non positive definite A"
            
            
    # CC^Tx = b
    z = np.zeros(p)
    x = np.zeros(p)
    # Forward C^Tx = z
    
    for i in range(p):
        sum = 0
        for j in range(i):
            sum += C[i, j] * z[j]
            
        z[i] = (b[i]-sum) / C[i, i]
        
    # Backward Cz = b
    for i in reversed(range(p)):
        sum = 0
        for j in range(i,p):
            sum += C[j, i] * x[j]
            
        x[i] = (z[i]-sum) / C[i, i]
        
    return x, mu

def soft_linesearch(xi0, h, gradient, phi, yhat, t, params, n, p, N, Q, delta, kmax, *args):
    print('doing line search')
    rho = 0.001
    sigma = 0.5
    tau1 = 9.0
    tau2 = 0.1
    tau3 = 0.5
    dxi0 = h @ gradient
    
    if dxi0 >= -10 * np.finfo(float).eps * LA.norm(h, 2) * LA.norm(gradient, 2):
        return 0
    
    mu = -xi0 / (rho * dxi0)

    alpha_old = 0
    alpha = min(1, mu)
    xi_old = xi0
    dxi_old = dxi0
    
    func_eval = 0
    # bracketing phase
    do_sectioning = True
    isvalid_xi_b = True
    print('soft_linesearch: start bracketing')
    while True:       
        # special treatment for non-negative parameters
        # if any of parameters is negative, then alpha should be reduced
        if any(params + alpha * h < 0):
            print('soft_linesearch: some of parameters are negative, alpha is too big, finishing bracketing')
            a = 0
            b = alpha
            xi_a = xi0
            dxi_a = dxi0
            tau3 = 0.1
            isvalid_xi_b = False
            break
        
        func_eval += 1
        xi_alpha, Y, hessian, gradient, _, fail = get_fYHg_stepwise(xi0, phi, yhat, t, params + alpha * h, n, p, N, Q, delta, *args)
        if fail: # integration failed, so alpha is big enough
            print('soft_linesearch: integration failed, alpha is too big, finishing bracketing')
            a = 0
            b = alpha
            xi_a = xi0
            dxi_a = dxi0
            tau3 = 0.1
            isvalid_xi_b = False
            break
        else:
            if xi_alpha < 1e-8:
                print('soft_linesearch: objective function is small enough')
                do_sectioning = False
                break
            
            if xi_alpha > xi0 + rho * dxi0 * alpha or xi_alpha > xi_old:
                a = alpha_old
                b = alpha
                xi_a = xi_old
                xi_b = xi_alpha
                dxi_a = dxi_old
                print('soft_linesearch: interval [',xi_a,',',xi_b,'] by xi_alpha')
                break
                
            dxi_alpha = h @ gradient
            
            if abs(dxi_alpha) <= -sigma * dxi0:
                print('soft_linesearch: gradient is small enough')               
                do_sectioning = False
                break
                
            if dxi_alpha >= 0: # right side of minimum
                a = alpha
                b = alpha_old
                xi_a = xi_alpha
                xi_b = xi_old
                dxi_a = dxi_alpha
                dxi_b = dxi_old
                print('soft_linesearch: interval [',xi_a,',',xi_b,'] by dxi_alpha')
                break
                
            da = alpha - alpha_old
            if mu <= alpha + da:
                alpha_old = alpha
                alpha = mu
                xi_old = xi_alpha
                dxi_old = dxi_alpha
            else:
                temp = alpha
                alpha = cubic_interpolate(alpha+da, min(mu, alpha+tau1*da), alpha_old, alpha, xi_old, xi_alpha, dxi_old, dxi_alpha)
                alpha_old = temp
                xi_old = xi_alpha
                dxi_old = dxi_alpha                                                                                 
    # sectioning phase
    if do_sectioning:
        print('soft_linesearch: start sectioning')
    while True and do_sectioning:       
        # special treatment for non-negative parameters
        # if any of parameters is negative, then alpha should be reduced
        while True:
            if any(params + b * h < 0):
                b *= 0.4
            else:
                break
        if not isvalid_xi_b:
            stop = False
            while not stop:
                func_eval += 1
                xi_b, Y, hessian, gradient, _, fail = get_fYHg_stepwise(xi0, phi, yhat, t, params + b * h, n, p, N, Q, delta, *args)        
                if fail: # b should be reduced
                    b *= 0.5
                    print('soft_linesearch: integration failed, try with b=%5.3f again' %b)
                else:
                    dxi_b = h @ gradient
                    if dxi_b < sigma * dxi0:
                        b *= 1.5
                        while True:
                            if any(params + b * h < 0):
                                b *= 0.95
                            else:
                                break
                        print('soft_linesearch: b is too small, increase b=%5.3f' %b)
                    else:
                        isvalid_xi_b = True
                        stop = True
        
        if abs(dxi_b) <= -sigma * dxi0:
            alpha = b
            xi_alpha = xi_b
            print('soft_linesearch: gradient is small enough')
            break
        d = b - a
        if d > 0:
            alpha = quadratic_interpolate(a+tau2*d, b-tau3*d, a, b, xi_a, xi_b, dxi_a)
        else:
            alpha = quadratic_interpolate(b-tau3*d, a+tau2*d, b, a, xi_a, xi_b, dxi_a)
        
        func_eval += 1
        xi_alpha, Y, hessian, gradient, _, fail = get_fYHg_stepwise(xi0, phi, yhat, t, params + alpha * h, n, p, N, Q, delta, *args)
        if fail:
            print('soft_something wrong')
            break
            
        if xi_alpha > xi0 + rho * dxi0 * alpha or xi_alpha > xi_a:
            b = alpha
            xi_b = xi_alpha
        else:
            dxi_alpha = h @ gradient
            if abs(dxi_alpha) <= -sigma * dxi0:
                break
            if (b - a) * dxi_alpha >= 0:
                b = a
                a = alpha
            else:
                a = alpha
        
    params += alpha * h
    
    print('soft_linesearch: line search done with %d evaluations.' %func_eval)
    return xi_alpha, params, Y, hessian, gradient, func_eval

def cubic_interpolate(x1, x2, a, b, fa, fb, dfa, dfb):
    cp = CubicHermiteSpline([a, b], [fa, fb], [dfa, dfb])
    fit = minimize(cp, x0=a, bounds=((x1,x2),))
    return fit.x[0]

def quadratic_interpolate(x1, x2, a, b, fa, fb, dfa):
    def qp(x):
        return fa + dfa * (x - a) + (fb - fa - (b - a) * dfa) * (x - a)**2 / (b - a)**2
    fit = minimize(qp, x0=a, bounds=((x1,x2),))
    return fit.x[0]

def bisection(xi0, h, gradient, phi, yhat, t, params, n, p, N, Q, delta, c=True, *args):
    print('doing bisection')
    dxi0 = h @ gradient
    
    if dxi0 >= -10 * np.finfo(float).eps * LA.norm(h, 2) * LA.norm(gradient, 2):
        return 0

    mu = 1.0

    while c:
        if not constraint(params + mu * h):
            mu /= 2.0
        else:
            break
        
    func_eval = 0
    while True:             
        func_eval += 1
        xi_mu, Y, hessian, gradient, _, fail = get_fYHg_stepwise(xi0, phi, yhat, t, params + mu * h, n, p, N, Q, delta, *args)
        if fail: # integration failed, so mu is too big
            mu /= 2.0
        else:
            break
     
    print('bisection done with mu={} and {} function evaluations.'.format(mu, func_eval))
    return xi_mu, params + mu * h, Y, hessian, gradient, mu, func_eval

    
def constraint(k):
    k0_lb = 0.0
    k1_lb = 0.0
    k1_ub = 1.0
    k2_lb = 0.0
    k2_ub = 1.0
    k3_lb = 0.0
    k3_ub = 1.0
    k4_lb = 0.0
    k5_lb = 0.0
    k5_ub = 500.0
    
    bool0 = k[0] > k0_lb
    bool1 = k[1] > k1_lb and k[1] < k1_ub
    bool2 = k[2] > k2_lb and k[2] < k2_ub
    bool3 = 1 - k[1] - k[2] > k3_lb and 1 - k[1] - k[2] < k3_ub
    # for critical model
    # bool4 = k[3] > k4_lb
    # bool5 = k[4] > k5_lb and k[4] < k5_ub
    
    res = bool0 and bool1 and bool2 and bool3
    # res = bool0 and bool1 and bool2 and bool3 and bool4 and bool5
    
    return res

def temp():
                
    # quadratic interpolation
    mu_opt = quadratic_interpolate(0, 2*mu, 0, mu, xi0, xi_mu, dxi0)
    func_eval += 1
    xi_opt, Y_opt, hessian_opt, gradient_opt, _, fail = get_fYHg_stepwise(xi0, phi, yhat, t, params + mu_opt * h, n, p, N, Q, delta, *args)
    if xi_opt < xi_mu:
        params += mu_opt * h
        print('bisection: bisection done with %d evaluations.' %func_eval)
        return xi_opt, params, Y_opt, hessian_opt, gradient_opt, mu, func_eval
    else:
        params += mu * h
        print('bisection: bisection done with %d evaluations.' %func_eval)
        return xi_mu, params, Y, hessian, gradient, mu, func_eval