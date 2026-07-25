// This kernel performs: output= std input−mean for every pixel.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void normalize_kernel(

    const float* input,

    float* output,

    int total_elements,

    float mean,

    float std

)

{


    int idx =

        blockIdx.x *
        blockDim.x
        +
        threadIdx.x;



    if(idx < total_elements)

    {


        output[idx] =

            (input[idx] - mean)
            /
            std;


    }

}




torch::Tensor normalize_cuda(

    torch::Tensor input,

    float mean,

    float std

)

{


    auto output = torch::empty_like(
        input
    );


    int total_elements =

        input.numel();



    int threads = 256;


    int blocks =

        (total_elements + threads - 1)
        /
        threads;



    normalize_kernel<<<

        blocks,

        threads

    >>>(

        input.data_ptr<float>(),

        output.data_ptr<float>(),

        total_elements,

        mean,

        std

    );



    return output;

}






torch::Tensor normalize_cpu(

    torch::Tensor input,

    float mean,

    float std

)

{


    auto output = torch::empty_like(
        input
    );


    auto input_accessor =
        input.accessor<float,4>();


    auto output_accessor =
        output.accessor<float,4>();



    for(
        int b=0;
        b<input.size(0);
        b++
    )

    {


        for(
            int c=0;
            c<input.size(1);
            c++
        )

        {


            for(
                int h=0;
                h<input.size(2);
                h++
            )

            {


                for(
                    int w=0;
                    w<input.size(3);
                    w++
                )

                {


                    output_accessor[b][c][h][w]

                    =

                    (
                    input_accessor[b][c][h][w]
                    -
                    mean
                    )
                    /
                    std;


                }

            }

        }

    }



    return output;

}
