// Native C++ extension source placeholder. This is the bridge between Python and CUDA.Python cannot directly call: __global__ void kernel() So we create bindings.
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
