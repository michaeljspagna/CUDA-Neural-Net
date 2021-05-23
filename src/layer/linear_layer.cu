#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_layer.hpp"

__global__ auto d_forward_propagation(float* weights
                                     ,float* A
                                     ,float* Z
                                     ,float* bias
                                     ,int weights_x
                                     ,int weights_y
                                     ,int A_x
                                     ,int A_y) -> void
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x = A_x;
    int Z_y = weights_y;
    float Z_value = 0;

    if (row < Z_y && col < Z_x) {
        for (int i = 0; i < weights_x; i++) {
            Z_value += weights[row * weights_x + i] * A[i * A_x + col];
        }
        Z[row * Z_x + col] = Z_value + bias[row];
    }
}

__global__ auto d_backward_propagation(float* weights
                                      ,float* Z_error
                                      ,float *A_error
                                      ,int weights_x
                                      ,int weights_y
                                      ,int Z_error_x
                                      ,int Z_error_y) -> void
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int A_error_x = Z_error_x;
    int A_error_y = weights_x;
    float A_error_value = 0.0f;

    if (row < A_error_y && col < A_error_x) {
        for (int i = 0; i < weights_y; i++) {
            A_error_value += weights[i * weights_x + row] * Z_error[i * Z_error_x + col];
        }
        A_error[row * A_error_x + col] = A_error_value;
    }
}

__global__ auto d_update_weights(float* Z_error
                                ,float* A
                                ,float* weights
                                ,int Z_error_x
                                ,int Z_error_y
                                ,int A_x
                                ,int A_y
                                ,float learning_rate) -> void
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int weights_x = A_y;
    int weights_y = Z_error_y;
    float weight_error_value = 0.0f;

    if (row < weights_y && col < weights_x) {
        for (int i = 0; i < Z_error_x; i++) {
            weight_error_value += Z_error[row * Z_error_x + i] * A[col * A_x + i];
        }
        weights[row * weights_x + col] = weights[row * weights_x + col] - learning_rate * (weight_error_value / A_x);
    }
}

__global__ auto d_update_bias(float* Z_error
                             ,float* bias
                             ,int Z_error_x
                             ,int Z_error_y
                             ,int bias_x
                             ,float learning_rate) -> void
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_error_x * Z_error_y) {
        int Z_error_x_adj = index % Z_error_x;
        int Z_error_y_adj = index / Z_error_x;
        atomicAdd(&bias[Z_error_y_adj], - learning_rate * (Z_error[Z_error_y_adj * Z_error_x + Z_error_x_adj] / Z_error_x));
    }
}

LinearLayer::LinearLayer(std::string title, Dimension weights_dimension)
    : weights(weights_dimension), bias(weights_dimension.y,1)
{
    this->title = title;
	bias.allocate_memory();
	weights.allocate_memory();
	initialize_bias();
	initalize_weights();
}

auto LinearLayer::forward(Tensor& A) -> Tensor&
{
	assert(weights.dimension.x == A.dimension.y);
	this->A = A;
	Dimension Z_dimension(A.dimension.x, weights.dimension.y);
	Z.allocate_memory_if_not_allocated(Z_dimension);
	compute_output(A);
	return Z;
}

auto LinearLayer::backprop(Tensor& Z_error, float learning_rate) -> Tensor&
{
    A_error.allocate_memory_if_not_allocated(A.dimension);
	compute_error(Z_error);
	update_bias(Z_error, learning_rate);
	update_weights(Z_error, learning_rate);
	return A_error;
}

auto LinearLayer::get_weights_x() const -> int
{
    return weights.dimension.x;
}

auto LinearLayer::get_weights_y() const -> int
{
    return weights.dimension.y;
}

auto LinearLayer::get_weights() const -> Tensor
{
    return weights;
}

auto LinearLayer::get_bias() const -> Tensor
{
    return bias;
}

auto LinearLayer::initialize_bias() -> void
{
    for (int x = 0; x < bias.dimension.x; x++) {
		bias[x] = 0;
	}

	bias.copy_host_to_device();
}

auto LinearLayer::initalize_weights() -> void
{
    std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < weights.dimension.x; x++) {
		for (int y = 0; y < weights.dimension.y; y++) {
			weights[y * weights.dimension.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	weights.copy_host_to_device();
}

auto LinearLayer::compute_error(Tensor& Z_error) -> void
{
	dim3 block_size(8, 8);
	dim3 block_count((A.dimension.x + block_size.x - 1) / block_size.x
					,(A.dimension.y + block_size.y - 1) / block_size.y);
	d_backward_propagation<<<block_count, block_size>>>(weights.device_data.get()
                                                       ,Z_error.device_data.get()
                                                       ,A_error.device_data.get()
                                                       ,weights.dimension.x
                                                       ,weights.dimension.y
                                                       ,Z_error.dimension.x
                                                       ,Z_error.dimension.y);
}

auto LinearLayer::compute_output(Tensor& A) -> void
{
    dim3 block_size(8, 8);
	dim3 block_count((Z.dimension.x + block_size.x - 1) / block_size.x
					,(Z.dimension.y + block_size.y - 1) / block_size.y);
	d_forward_propagation<<<block_count, block_size>>>(weights.device_data.get()
                                                      ,A.device_data.get()
                                                      ,Z.device_data.get()
                                                      ,bias.device_data.get()
                                                      ,weights.dimension.x
                                                      ,weights.dimension.y
                                                      ,A.dimension.x
                                                      ,A.dimension.y);
}

auto LinearLayer::update_weights(Tensor& Z_error, float learning_rate) -> void
{
	dim3 block_size(8, 8);
	dim3 block_count((weights.dimension.x + block_size.x - 1) / block_size.x
					  ,(weights.dimension.y + block_size.y - 1) / block_size.y);
    d_update_weights<<<block_count, block_size>>>(Z_error.device_data.get()
                                                 ,A.device_data.get()
                                                 ,weights.device_data.get()
                                                 ,Z_error.dimension.x
                                                 ,Z_error.dimension.y
                                                 ,A.dimension.x
                                                 ,A.dimension.y
                                                 ,learning_rate);
}

auto LinearLayer::update_bias(Tensor& Z_error, float learning_rate) -> void
{
    dim3 block_size(256);
	dim3 block_count( (Z_error.dimension.y * Z_error.dimension.x + block_size.x - 1) / block_size.x);
	d_update_bias<<<block_count, block_size>>>(Z_error.device_data.get()
                                              ,bias.device_data.get()
                                              ,Z_error.dimension.x
                                              ,Z_error.dimension.y
                                              ,bias.dimension.x
                                              ,learning_rate);
}

int main(){}