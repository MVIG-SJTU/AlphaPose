# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import numpy as np
import os
from os.path import join as pjoin
#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess

#change for windows, by MrX
nvcc_bin = 'nvcc.exe'
lib_dir = 'lib/x64'

import distutils.msvc9compiler
distutils.msvc9compiler.VERSION = 14.0


def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        print("home = %s\n" % home)
        nvcc = pjoin(home, 'bin', nvcc_bin)
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path(nvcc_bin, os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_PATH')
        home = os.path.dirname(os.path.dirname(nvcc))
        print("home = %s, nvcc = %s\n" % (home, nvcc))


    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, lib_dir)}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    #self.src_extensions.append('.cu')

	
    # save references to the default compiler_so and _comple methods
    #default_compiler_so = self.spawn 
    #default_compiler_so = self.rc
    super = self.compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
        postfix=os.path.splitext(sources[0])[1]
        
        if postfix == '.cu':
            # use the cuda for .cu files
            #self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']


        return super(sources, output_dir, macros, include_dirs, debug, extra_preargs, postargs, depends)
        # reset the default compiler_so, which we might have changed for cuda
        #self.rc = default_compiler_so

    # inject our redefined _compile method into the class
    self.compile = compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    # unix _compile: obj, src, ext, cc_args, extra_postargs, pp_opts
    Extension(
        "cpu_nms",
        sources=["cpu_nms.pyx"],
        extra_compile_args={'gcc': []},
        include_dirs = [numpy_include],
    ),
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)
