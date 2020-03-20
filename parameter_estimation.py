import numpy as np

def lm_breakage(breakage, bfunc, Sfunc, yhat, L, Q, k0, t, opts = [1e-3, 1e-8, 1e-8, 1000]):
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
            
        p = np.size(k0)
        k = k0.copy()
        dbs = discretize(bfunc, Sfunc, L, n, p, k, delta)
        Y, Jt, S, r, fail = checkSrJ(breakage, yhat, dbs, t, n, p, N, Q, 1e8, scalar, delta)
        assert not fail, "Huge residuals"
        S0 = S
        r0 = r.copy()
        H, g = Hg(Jt, Q, r, p, N)
        gn = LA.norm(g, np.inf)
        stop = bool(gn < opts[1])
        
        if stop:
            ter = 'g'
            print("First guess was correct!")
            
        mu = opts[0] * max(np.diag(H))
        print('Iter | Obj func | step size | gradient |   mu   |   rho')
        print("{0:5d}|{1:10.4e}|   Not cal |  Not cal |{2:8.1e}| Not cal"
              .format(it,S,mu))
        
        while (not stop) and (it <= opts[3]):
            fail = False
            it += 1
            h, mu = cholsolve(H, -g, mu, p)
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if hn <= opts[2] * (kn + opts[2]):
                stop = True
                ter = 'h'
            else:
                k_new = k + h
                dbs = discretize(bfunc, Sfunc, L, n, p, k_new, delta)
                Y, Jt, S, r, fail = checkSrJ(breakage, yhat, dbs, t, n,
                                             p, N, Q, S0, scalar, delta)
                dL = h @ (mu * h - g) / 2
                
                if dL > 0 and not fail:
                    df = dF(r0, r, Q, n, N)
                    rho = df / dL
                    k = k_new 
                    S0 = S
                    r0 = r.copy()
                    H, g = Hg(Jt, Q, r, p, N)
                    gn = LA.norm(g, np.inf) 
                    
                    if gn < opts[1]:
                        stop = True
                        ter = 'g'
                    mu *= max(1/3,1-(2*rho-1)**3)
                    nu = 2
                else:
                    mu *= nu
                    nu *= 2
                    
            if rho == 0:
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}| Not cal"
                        .format(it,S,hn,gn,mu))
            else:
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}|{5:8.1e}"
                        .format(it,S,hn,gn,mu,rho))
                
        info = [it, ter]
        output = [k, Y, dbs[0], dbs[1], info]
        return output
    
    except OverflowError as oerr:
        print(oerr.args)
        return
    
    except AssertionError as aerr:
        print(aerr.args)
        return   

 

def lm_breakage_red(breakage, bfunc, Sfunc, yhat, L, Q, k0, t, opts = [1e-3, 1e-8, 1e-8, 1000]):
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
            
        p = np.size(k0)
        k = k0.copy()
        dbs = discretize(bfunc, Sfunc, L, n, p, k, delta)
        Y, Jt, S, r, fail = checkSrJ(breakage, yhat, dbs, t, n, p, N, Q, 1e8, scalar, delta)
        assert not fail, "Huge residuals"
        S0 = S
        r0 = r.copy()
        H, g = Hg(Jt, Q, r, p, N)
        gn = LA.norm(g, np.inf)
        stop = bool(gn < opts[1])
        
        if stop:
            ter = 'g'
            print("First guess was correct!")
            
        mu = opts[0] * max(np.diag(H))
        print('Iter | Obj func | step size | gradient |   mu   |   rho')
        print("{0:5d}|{1:10.4e}|   Not cal |  Not cal |{2:8.1e}| Not cal"
              .format(it,S,mu))
        
        while (not stop) and (it <= opts[3]):
            fail = False
            it += 1
            
            K = np.diag(k)
            Hr = K @ H @ K
            gr = K @ g
            
            hr, mu = cholsolve(Hr, -gr, mu, p)
            h = K @ hr
            
            hn = LA.norm(h, 2)
            kn = LA.norm(k, 2)
            
            if hn <= opts[2] * (kn + opts[2]):
                stop = True
                ter = 'h'
            else:
                k_new = k + h
                dbs = discretize(bfunc, Sfunc, L, n, p, k_new, delta)
                Y, Jt, S, r, fail = checkSrJ(breakage, yhat, dbs, t, n,
                                             p, N, Q, S0, scalar, delta)
                dL = h @ (mu * h - g) / 2
                
                if dL > 0 and not fail:
                    df = dF(r0, r, Q, n, N)
                    rho = df / dL
                    k = k_new 
                    S0 = S
                    r0 = r.copy()
                    H, g = Hg(Jt, Q, r, p, N)
                    gn = LA.norm(g, np.inf) 
                    
                    if gn < opts[1]:
                        stop = True
                        ter = 'g'
                    mu *= max(1/3,1-(2*rho-1)**3)
                    nu = 2
                else:
                    mu *= nu
                    nu *= 2
                    
            if rho == 0:
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}| Not cal"
                        .format(it,S,hn,gn,mu))
            else:
                print("{0:5d}|{1:10.4e}|{2:11.3e}|{3:10.2e}|{4:8.1e}|{5:8.1e}"
                        .format(it,S,hn,gn,mu,rho))
                
        info = [it, ter]
        output = [k, Y, dbs[0], dbs[1], info]
        return output
    
    except OverflowError as oerr:
        print(oerr.args)
        return
    
    except AssertionError as aerr:
        print(aerr.args)
        return   



def checkSrJ(breakage, yhat, dbs, t, n, p, N, Q, S0, scalar, delta):
    # initial condition J0 = 0
    if scalar:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
        
    r = np.zeros((n,N))    
    Z = np.zeros((n*(p+1),N))
    Z[0:n,0] = y0.copy()
    S = 0
    i = 0
    fail = False
    
    print('start loop')
    while (not fail) and i < N-1:
        print(i)
        Z[:, i+1], suc = integ_breakage_onestep(breakage, Z[:,i], dbs,
                                                [t[i], t[i+1]], n, p, delta)
        
        if not suc:
            S = S0
            fail = True
        else:
            if scalar:            
                r[:, i+1] = yhat[i+1] - Z[0, i+1]
            else:
                r[:, i+1] = yhat[:, i+1]-Z[0:n, i+1] 
                
            S += r[:,i+1] @ Q @ r[:,i+1] / 2
            
            if S>S0:
                S = S0
                fail = True
                
        i += 1
        
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    return Y, Jt, S, r, fail    



def Hg(Jt, Q, r, p, N):
    H = np.zeros((p, p))
    g = np.zeros(p)
    
    for i in range(N):
        JQ = Jt[i].T @ Q
        H += JQ @ Jt[i]
        g -= JQ @ r[:,i]
        
    return H,g


def dF(r0, r , Q, n, N):
    dS = 0
    
    if n == 1:
        dS = (r0[0] - r[0]) @ (r0[0] + r[0]) / 2
    else:
        for i in range(N):
            dS += (r0[:,i] - r[:,i]) @ Q @ (r0[:,i] + r[:,i]) /2
            
    return dS
   
    
    
def cholesky(A, p):
    C = np.zeros((p,p))
    j = 0
    pd = True
    
    while pd and j < p:
        sum = 0
        
        for k in range(j):
            sum += C[j][k]**2
            
        d = A[j][j]-sum
        
        if d>0:
            C[j][j] = np.sqrt(d)
            
            for i in range(j,p):
                sum = 0
                for k in range(j):
                    sum += C[i][k] * C[j][k]
                C[i][j] = (A[i][j]-sum) / C[j][j]
                
        else:
            pd = False
            
        j += 1
        
    return C,pd



def cholsolve(A, b, mu, p):
    I = np.eye(p)
    mA = np.amax(abs(A))
    pd = False
    
    while pd == False:
        C,pd = cholesky(A + mu * I, p)
        
        # check for near singularity
        if pd == True:
            pd = (1 / LA.cond(C,1) >= 1e-15)
        if pd == False:
            mu = max(10 * mu, np.finfo(float).eps * mA)
            
    # CC^Tx = b
    z = np.zeros(p)
    x = np.zeros(p)
    # Forward C^Tx = z
    
    for i in range(p):
        sum = 0
        for j in range(i):
            sum += C[i][j] * z[j]
            
        z[i] = (b[i]-sum) / C[i][i]
        
    # Backward Cz = b
    for i in reversed(range(p)):
        sum = 0
        for j in range(i,p):
            sum += C[j][i] * x[j]
            
        x[i] = (z[i]-sum) / C[i][i]
        
    return x,mu