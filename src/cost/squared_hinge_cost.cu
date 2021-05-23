#include "squared_hinge_cost.hpp"

#include <cassert>
#include <cmath>

__global__ auto d_cost(float* predictions
                      ,float* target
                      ,int size
                      ,float* cost) -> void
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

auto SquaredHingeCost::cost(Tensor predictions, Tensor target) -> float
{
	float* cost;
	return cost_value;
}

auto SquaredHingeCost::compute_gradient(Tensor predictions, Tensor target, Tensor gradient) -> Tensor
{
    return gradient;
}

int main(){}