from distutils.core import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("lognormal_cy", ["lognormal_cy.pyx"],),
    Extension("ode", ["ode.pyx"],),
    Extension("phi_cy", ["phi_cy.pyx"],),
    Extension("ode_nogil", ["ode_cy_nogil.pyx"]),
    Extension("phi_parallel", ["phi_cy_parallel.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],)
]

setup(name="PBM",
      ext_modules=cythonize(extensions),)

