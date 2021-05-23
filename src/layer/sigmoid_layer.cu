#include "sigmoid_layer.hpp"

__device__ auto d_sigmoid_activation(float x) -> float
{
    return 1.0f / (1 + expf(-x));
}

__global__ auto d_forward_propagation(float* Z
                                     ,float* A
                                     ,int Z_x
                                     ,int Z_y) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx < (Z_x * Z_y))
        A[thread_idx] = d_sigmoid_activation(Z[thread_idx]);
}

__global__ auto d_backward_propagation(float* Z
                                      ,float* A_error
                                      ,float* Z_error
                                      ,int Z_x
                                      ,int Z_y) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < (Z_x * Z_y))
        Z_error[thread_idx] = A_error[thread_idx] * d_sigmoid_activation(Z[thread_idx]) * (1 - d_sigmoid_activation(Z[thread_idx]));
}

SigmoidLayer::SigmoidLayer(std::string title)
{
    this->title = title;
}

auto SigmoidLayer::forward_propagation(Tensor& Z) -> Tensor&
{
    this->Z = Z;
    A.allocate_memory_if_not_allocated(Z.dimension);
    dim3 block_size(256);
    dim3 block_count((Z.dimension.x * Z.dimension.y + block_size.x - 1) / block_size.x);
    d_forward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                      ,A.device_data.get()
                                                      ,Z.dimension.x
                                                      ,Z.dimension.y);
    return A;
}

auto SigmoidLayer::backward_propagation(Tensor& A_error, float learning_rate) -> Tensor&
{
    Z_error.allocate_memory_if_not_allocated(Z.dimension);\
    dim3 block_size(256);
    dim3 block_count((Z.dimension.x * Z.dimension.y + block_size.x - 1) / block_size.x);
    d_backward_propagation<<<block_count, block_size>>>(Z.device_data.get()
                                                       ,A_error.device_data.get()
                                                       ,Z_error.device_data.get()
                                                       ,Z.dimension.x
                                                       ,Z.dimension.y);          
    return Z_error;
}

int main(){return 1;}