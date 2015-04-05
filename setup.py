
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
ext = Extension(name = "kb2vec_inner",sources=["kb2vec_inner.pyx"])

setup(name="kb2vec_inner",ext_modules = cythonize([ext]),
      include_dirs=[numpy.get_include()])
