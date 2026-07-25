import time

import torch

from dataset import create_dataloaders

import config



# Import your compiled CUDA extension
#
# This assumes after compiling:
#
# native_ops/
#       setup.py
#       native_ops.cpp
#       normalize_kernel.cu
#
# creates:
#
# import native_ops


try:

    import native_ops

    CUDA_EXTENSION_AVAILABLE = True


except ImportError:

    print(
        "Warning: native_ops not found."
    )

    CUDA_EXTENSION_AVAILABLE = False





def benchmark_cpu(images):
    """
    Benchmarks CPU preprocessing.

    Uses OpenMP implementation
    inside your C++ extension.

    """


    start = time.perf_counter()


    output = native_ops.normalize_images(
        images,
        config.CUDA_MEAN,
        config.CUDA_STD,
        False
    )


    end = time.perf_counter()


    elapsed = (
        end - start
    )


    return elapsed, output





def benchmark_gpu(images):
    """
    Benchmarks CUDA preprocessing.

    Steps:

        CPU tensor

             |
             |
        cudaMemcpy

             |
             |
        CUDA kernel

             |
             |
        GPU tensor


    """


    images = images.cuda(
        non_blocking=True
    )


    # Synchronize before timing

    torch.cuda.synchronize()


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    start.record()



    output = native_ops.normalize_images(
        images,
        config.CUDA_MEAN,
        config.CUDA_STD,
        True
    )



    end.record()



    torch.cuda.synchronize()


    elapsed = (
        start.elapsed_time(end)
        /
        1000
    )


    return elapsed, output





def run_benchmark():

    """
    Runs benchmark using
    real EuroSAT images.

    NOT random tensors.

    """


    if not CUDA_EXTENSION_AVAILABLE:

        raise RuntimeError(
            "Compile native_ops first."
        )



    print(
        "Loading EuroSAT batch..."
    )


    train_loader, _, _ = (
        create_dataloaders()
    )


    images, labels = next(
        iter(train_loader)
    )


    print(
        f"Batch shape: {images.shape}"
    )


    print(
        "Running CPU benchmark..."
    )


    cpu_time, cpu_output = (
        benchmark_cpu(images)
    )


    print(
        "Running CUDA benchmark..."
    )


    gpu_time, gpu_output = (
        benchmark_gpu(images)
    )



    speedup = (
        cpu_time /
        gpu_time
    )


    print("\n====================")

    print(
        f"CPU Time: {cpu_time:.6f}s"
    )

    print(
        f"GPU Time: {gpu_time:.6f}s"
    )

    print(
        f"CUDA Speedup: {speedup:.2f}x"
    )

    print(
        "====================\n"
    )



    return {

        "cpu_seconds":
            cpu_time,

        "cuda_seconds":
            gpu_time,

        "speedup":
            speedup

    }





if __name__ == "__main__":

    results = run_benchmark()


    print(results)
