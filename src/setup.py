from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_transforms',
    ext_modules=[
        CUDAExtension('cuda_transforms', [
            'transformations.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
