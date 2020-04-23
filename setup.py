from distutils.core import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("lognormal_cy", ["lognormal_cy.pyx"],),
    Extension("lognormal_cy_td", ["lognormal_cy_td.pyx"],),
    Extension("lognormal_cy_critical", ["lognormal_cy_critical.pyx"],),
    Extension("ode_cy", ["ode_cy.pyx"],),
    Extension("ode_cy_cdef", ["ode_cy_cdef.pyx"],),
    Extension("ode_cy_parallel", ["ode_cy_parallel.pyx"],),
    Extension("discretize_cy", ["discretize_cy.pyx"]),
    Extension("phi_cy", ["phi_cy.pyx"],),
    Extension("ode_nogil", ["ode_nogil.pyx"]),
    Extension("phi_cy_parallel", ["phi_cy_parallel.pyx"],)
]

setup(name="PBM",
      ext_modules=cythonize(extensions),)