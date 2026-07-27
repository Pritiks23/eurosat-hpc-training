""" This setup.py file is responsible for compiling the custom C++ and CUDA source files into a Python extension that PyTorch can import. 
Python cannot execute C++ or CUDA code directly, so this build step creates a shared library (native_ops) that acts like a normal 
Python module. It includes both the C++ binding file (native_ops.cpp) and the CUDA kernel (normalize_kernel.cu), while BuildExtension
uses PyTorch's build system to compile, link, and configure the extension correctly for the installed CUDA and PyTorch versions.
Without this file, the optimized native code could not be called from the Python training pipeline."""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="native_ops",
    ext_modules=[
        CUDAExtension(
            name="native_ops",
            sources=[
                "native_ops.cpp",
                "normalize_kernel.cu"
            ],
        )
    ],
    cmdclass={
        "BuildExtension": BuildExtension
    }
)
