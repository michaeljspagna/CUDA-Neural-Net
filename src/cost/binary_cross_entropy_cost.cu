#include "binary_cross_entropy_cost.hpp"

#include <cassert>
#include <cmath>

__global__ auto d_cost(float* predictions
                      ,float* target
                      ,int size
                      ,float* cost) -> void
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float partial_cost = target[index] * logf(predictions[index])
                           + (1.0f - target[index]) * logf(1.0f - predictions[index]);
        atomicAdd(cost, - partial_cost / size);
    }
}

__global__ auto d_compute_gradient(float* predictions
                                  ,float* target
                                  ,float* gradient
                                  ,int size) -> void
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        gradient[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
    }
}

auto BinaryCrossEntropyCost::cost(Tensor predictions, Tensor target) -> float
{
	assert(predictions.dimension.x == target.dimension.x);
	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;
	dim3 block_size(256);
	dim3 block_count((predictions.dimension.x + block_size.x - 1) / block_size.x);
	d_cost<<<block_count, block_size>>>(predictions.device_data.get()
                                       ,target.device_data.get()
                                       ,predictions.dimension.x
                                       ,cost);
	cudaDeviceSynchronize();
	float cost_value = *cost;
	cudaFree(cost);
	return cost_value;
}

auto BinaryCrossEntropyCost::compute_gradient(Tensor predictions, Tensor target, Tensor gradient) -> Tensor
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

int main(){}