from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='nms_cpu',
    ext_modules=[
        CppExtension(
            name='nms_cpu',
            sources=[
                'src/nms_cpu.cpp',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)