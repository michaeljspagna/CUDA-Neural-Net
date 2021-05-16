#include "relu_activation_layer.cuh"

__global__ auto ReLUActivationLayer::relu_forward_propagation(float* Z
                                        ,float* A
                                        ,Dimensions Z_dim) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx < (int)(Z_dim.x*Z_dim.y))
        A[thread_idx] = fmaxf(Z[thread_idx], 0);
}
__global__ auto ReLUActivationLayer::relu_backward_propagation(float* Z
                                         ,float* dA
                                         ,float* dZ
                                         ,Dimensions Z_dim) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx < (int)(Z_dim.x*Z_dim.y))
    {
        if(Z[thread_idx] > 0){
            dZ[thread_idx] = dA[thread_idx];
        }else{
            dZ[thread_idx] = 0;
        }
    }
}

ReLUActivationLayer::ReLUActivationLayer(std::string title)
{
    this->title = title;
}

auto ReLUActivationLayer::forward_propagation(Tensor& Z) -> Tensor&
{
    this->Z = Z;
    A.allocate_if_not_allocated(Z.dimensions);

    dim3 block_size(256);
    dim3 block_count = ((Z.dimensions.y * Z.dimensions.x + block_size.x - 1) / block_size.x);

    relu_forward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                         ,A.device_data.get()
                                                         ,Z.dimensions);

    return A;
}

auto ReLUActivationLayer::backward_propagation(Tensor& dA, float alpha=0.01) -> Tensor&
{
    dZ.allocate_if_not_allocated(Z.dimensions);

    dim3 block_size(256);
    dim3 block_count = ((Z.dimensions.y * Z.dimensions.x + block_size.x - 1) / block_size.x);

    relu_backward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                          ,dA.device_data.get()
                                                          ,dZ.device_data.get()
                                                          ,Z.dimensions);

    return dZ;
}