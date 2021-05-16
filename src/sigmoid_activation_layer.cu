#include "sigmoid_activation_layer.cuh"

__device__ auto SigmoidActivationLayer::logistic_sigmoid(float x) -> float
{
    return 1.0f / (1 + expf(-x));
}

__global__ auto SigmoidActivationLayer::sigmoid_forward_propagation(float* Z
                                                                   ,float* A
                                                                   ,Dimensions Z_dim) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx < (int)(Z_dim.x*Z_dim.y))
        A[thread_idx] = logistic_sigmoid(Z[thread_idx]);
}

__global__ auto SigmoidActivationLayer::sigmoid_backward_propagation(float* Z
                                                                    ,float* dA
                                                                    ,float* dZ
                                                                    ,Dimensions Z_dim) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < (int)(Z_dim.x*Z_dim.y))
        dZ[thread_idx] = dA[thread_idx] * logistic_sigmoid(Z[thread_idx]) * (1 - logistic_sigmoid(Z[thread_idx]));
}

SigmoidActivationLayer::SigmoidActivationLayer(std::string title)
{
    this->title = title;
}

auto forward_propagation(Tensor& Z) -> Tensor&
{
    this->Z = Z;
    A.allocate_if_not_allocated(Z.dimensions);

    dim3 block_size(256);
    dim3 block_count((Z.dimensions.x * Z.dimensions.y + block_size.x - 1) / block_size.x);

    sigmoid_forward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                            ,A.device_data.get()
                                                            ,Z.dimensions);
    
    return A;
}

auto backward_propagation(Tensor& dA, float alpha=0.01) -> Tensor&
{
    dZ.allocate_if_not_allocated(Z.dimensions);\

    dim3 block_size(256);
    dim3 block_count((Z.dimensions.x * Z.dimensions.y + block_size.x - 1) / block_size.x);

    sigmoid_backward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                             ,dA.device_data.get()
                                                             ,dZ.device_data.get()
                                                             ,Z.dimensions);
                                                    
    return dZ;
}