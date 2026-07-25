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
