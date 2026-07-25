from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension
)


setup(

    name="native_ops",

    ext_modules=[

        CUDAExtension(

            name="native_ops",

            sources=[

                "native_ops.cpp",

                "normalize_kernel.cu"

            ],

            extra_compile_args={

                "cxx": [

                    "-O3"

                ],

                "nvcc": [

                    "-O3",

                    "--use_fast_math"

                ]

            }

        )

    ],


    cmdclass={

        "BuildExtension":
            BuildExtension

    }

)
