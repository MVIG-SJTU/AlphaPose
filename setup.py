import os
import platform
import subprocess
import time

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

MAJOR = 0
MINOR = 3
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'alphapose/version.py'


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from alphapose.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def get_ext_modules():
    ext_modules = []
    # only windows visual studio 2013+ support compile c/cuda extensions
    # If you force to compile extension on Windows and ensure appropriate visual studio
    # is intalled, you can try to use these ext_modules.
    force_compile = False
    if platform.system() != 'Windows' or force_compile:
        ext_modules = [
            make_cython_ext(
                name='soft_nms_cpu',
                module='detector.nms',
                sources=['src/soft_nms_cpu.pyx']),
            make_cuda_ext(
                name='nms_cpu',
                module='detector.nms',
                sources=['src/nms_cpu.cpp']),
            make_cuda_ext(
                name='nms_cuda',
                module='detector.nms',
                sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']),
            make_cuda_ext(
                name='roi_align_cuda',
                module='alphapose.utils.roi_align',
                sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu']),
            make_cuda_ext(
                name='deform_conv_cuda',
                module='alphapose.models.layers.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                module='alphapose.models.layers.dcn',
                sources=[
                    'src/deform_pool_cuda.cpp',
                    'src/deform_pool_cuda_kernel.cu'
                ]),
        ]
    return ext_modules


def get_install_requires():
    install_requires = [
        'six', 'terminaltables', 'scipy==1.1.0',
        'opencv-python', 'matplotlib', 'visdom',
        'tqdm', 'tensorboardx', 'easydict',
        'pyyaml',
        'torch>=1.1.0', 'torchvision>=0.3.0',
        'munkres', 'timm==0.1.20', 'natsort'
    ]
    # official pycocotools doesn't support Windows, we will install it by third-party git repository later
    if platform.system() != 'Windows':
        install_requires.append('pycocotools==2.0.0')
    return install_requires


def is_installed(package_name):
    from pip._internal.utils.misc import get_installed_distributions
    for p in get_installed_distributions():
        if package_name in p.egg_name():
            return True
    return False


if __name__ == '__main__':
    write_version_py()
    setup(
        name='alphapose',
        version=get_version(),
        description='Code for AlphaPose',
        long_description=readme(),
        keywords='computer vision, human pose estimation',
        url='https://github.com/MVIG-SJTU/AlphaPose',
        packages=find_packages(exclude=('data', 'exp',)),
        package_data={'': ['*.json', '*.txt']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='GPLv3',
        python_requires=">=3",
        setup_requires=['pytest-runner', 'numpy', 'cython'],
        tests_require=['pytest'],
        install_requires=get_install_requires(),
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
    # Windows need pycocotools here: https://github.com/philferriere/cocoapi#subdirectory=PythonAPI
    if platform.system() == 'Windows' and not is_installed('pycocotools'):
        print("\nInstall third-party pycocotools for Windows...")
        cmd = 'python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI'
        os.system(cmd)
    if not is_installed('cython_bbox'):
        print("\nInstall `cython_bbox`...")
        cmd = 'python -m pip install git+https://github.com/yanfengliu/cython_bbox.git'
        os.system(cmd)
