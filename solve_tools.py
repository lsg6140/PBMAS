import numpy as np
from numpy import linalg as LA

from solve_ode import evolve

def get_fYHg(phi, yhat, t, params, n, p, N, Q, scalar, delta, *args):
    # initial condition J0 = 0
    if scalar:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
        
    r = np.zeros((n,N))    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()
    f = 0
    fail = False    
    
    print('parameters are', params)
    Z, suc = evolve(phi, Z0, t, params, n, p, delta, *args)
    
    if not suc:
        print('solving ODE failed')
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

def soft_linesearch(f0, h, gradient, phi, yhat, t, params, n, p, N, Q, scalar, delta, amax, kmax, *args):
    print('doing line search')
    rho = 0.001
    sigma = 0.1
    df0 = h @ gradient
    
    if df0 >= -10 * np.finfo(float).eps * LA.norm(h, 2) * LA.norm(gradient, 2):
        return 0
    
    a = 0
    b = min(1, amax)
    
    iter = 0
    # bracketing phase
    while True:
        fb, Y, hessian, gradient, _, fail = get_fYHg(phi, yhat, t, params + b * h, n, p, N, Q, scalar, delta, *args)
        if fail: # integration failed
            b /= 5
            print('trying again with b=%5.3f' % b)
        else:
            dfb = h @ gradient
            if fb < f0 + rho * df0 * b and dfb < sigma * df0 and b < amax and iter < kmax:
                iter += 1
                a = b
                fa = fb
                dfa = dfb

                if 2.5 * b >= amax:
                    b = amax
                else:
                    b *= 2
            else:
                break
    
    alpha = b
    falpha = fb
    dfalpha = dfb
    # sectioning phase
    while True:
        if (falpha > f0 + rho * df0 * alpha or dfalpha < sigma * df0) and iter < kmax:
            print('doing refine')
            iter += 1
            alpha, falpha, a, b, fa, fb, dfa, Y, hessian, gradient = refine(a, b, fa, fb, f0, dfa, df0, h, rho, phi, yhat, t, params, n, p, N, Q, scalar, delta, *args)
        else:
            print('no need to refine')
            break
            
    if falpha > f0:
        alpha = 0
    
    params += alpha * h
            
    return falpha, params, Y, hessian, gradient


def refine(a, b, fa, fb, f0, dfa, df0, h, rho, phi, yhat, t, params, n, p, N, Q, scalar, delta, *args):
    D = b - a
    c = (fb - fa - D * dfa) / D**2

    if c > 0:
        alpha = a - dfa / (2 * c)
        alpha = min(max(alpha, a + 0.1 * D), b - 0.1 * D)
    else:
        alpha = (a + b) / 2

    falpha, Y, hessian, gradient, _, fail = get_fYHg(phi, yhat, t, params + alpha * h, n, p, N, Q, scalar, delta, *args)
    assert not fail, 'something wrong...'

    dfalpha = h @ gradient
    if falpha < f0 + rho * df0 * alpha:
        a = alpha
        fa = falpha
        dfa = dfalpha
    else:
        b = alpha
        fb = falpha
        dfb = dfalpha
        
    return alpha, falpha, a, b, fa, fb, dfa, Y, hessian, gradient