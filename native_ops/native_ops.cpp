// Native C++ extension source placeholder. This is the bridge between Python and CUDA.Python cannot directly call: __global__ void kernel() So we create bindings.
// This file acts as the bridge between Python and the high-performance C++/CUDA code, allowing your PyTorch training script to call native functions that Python cannot execute directly. It registers the normalize_images function as a PyTorch extension using PyBind11 and provides a single interface that automatically dispatches the request to either the CPU (OpenMP) or GPU (CUDA) implementation based on the use_cuda flag. This design hides the hardware-specific implementation details from the Python code, making the training pipeline simpler while still taking advantage of optimized native performance.
#include <torch/extension.h>


// CUDA function declaration

torch::Tensor normalize_cuda(
    torch::Tensor input,
    float mean,
    float std
);



torch::Tensor normalize_cpu(
    torch::Tensor input,
    float mean,
    float std
);




torch::Tensor normalize_images(

    torch::Tensor input,

    float mean,

    float std,

    bool use_cuda

)

{


    if(use_cuda)

    {

        return normalize_cuda(
            input,
            mean,
            std
        );

    }


    else

    {

        return normalize_cpu(
            input,
            mean,
            std
        );

    }

}




PYBIND11_MODULE(

    TORCH_EXTENSION_NAME,

    m

)

{

    m.def(

        "normalize_images",

        &normalize_images,

        "Image normalization CPU/CUDA"

    );

}
