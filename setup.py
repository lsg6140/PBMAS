from distutils.core import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("lognormal_cy", ["lognormal_cy.pyx"],),
]

setup(name="PBM",
      ext_modules=cythonize(extensions),)

