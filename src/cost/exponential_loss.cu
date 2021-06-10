#include "exponential_loss.hpp"

#include <cassert>
#include <cmath>

__global__ auto d_loss(float* predictions
                      ,float* target
                      ,int size
                      ,float* loss) -> void
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {

    }
}

__global__ auto d_compute_gradient(float* predictions
                                  ,float* target
                                  ,float* gradient
                                  ,int size) -> void
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
    }
}

auto ExponentialLoss::loss(Tensor predictions, Tensor target) -> float
{
	assert(predictions.dimension.x == target.dimension.x);
	float* loss;
	cudaMallocManaged(&loss, sizeof(float));
	*loss = 0.0f;
	dim3 block_size(256);
	dim3 block_count((predictions.dimension.x + block_size.x - 1) / block_size.x);
	d_loss<<<block_count, block_size>>>(predictions.device_data.get()
                                       ,target.device_data.get()
                                       ,predictions.dimension.x
                                       ,loss);
	cudaDeviceSynchronize();
	float loss_value = *loss;
	cudaFree(loss);
	return loss_value;
}

auto ExponentialLoss::compute_gradient(Tensor predictions, Tensor target, Tensor gradient) -> Tensor
{
    assert(predictions.dimension.x == target.dimension.x);

    dim3 block_size(256);
    dim3 block_count((predictions.dimension.x + block_size.x - 1)/ block_size.x);
    d_compute_gradient<<<block_count, block_size>>>(predictions.device_data.get()
                                                   ,target.device_data.get()
                                                   ,gradient.device_data.get()
                                                   ,predictions.dimension.x);
    return gradient;
}