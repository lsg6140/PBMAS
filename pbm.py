from discretize_parallel import discretize
from phi_cy_parallel import phi_breakage

def phi(z, t, params, n, p, delta, *args):
    length = args[0]
    mu = args[1]
    sigma = args[2]
    dbs = discretize(length, n, p, params, delta, mu, sigma)
    
    return phi_breakage(z, dbs, n, p, delta)