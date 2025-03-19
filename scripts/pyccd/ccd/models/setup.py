from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cython_lasso",
        ["cython_lasso.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="cython_lasso",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True
    }),
    include_dirs=[np.get_include()]
)
