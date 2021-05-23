#include <random>
#include "linear_activation_layer.cuh"

__global__ auto linear_forward_propagation(float* W
                                          ,float* A
                                          ,float* Z
                                          ,float* b
                                          ,int W_x,
                                          ,int W_y
                                          ,int A_x
                                          ,int A_y) -> void
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x = A_x;
	int Z_y = W_y;

	float Z_value = 0;

	if (row < Z_y && col < Z_x) {
		for (int i = 0; i < W_x; i++) {
			Z_value += W[row * W_x + i] * A[i * A_x + col];
		}
		Z[row * Z_x + col] = Z_value + b[row];
	}
}

__global__ auto linear_backward_propagation(float* W
                                           ,float* dZ
                                           ,float* dA
                                           ,int W_x
                                           ,int W_y
                                           ,int dZ_x
                                           ,int dZ_y) -> void
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int dA_x = dZ_x;
	int dA_y = W_x;

	float dA_value = 0.0f;

	if (row < dA_y && col < dA_x) {
		for (int i = 0; i < W_y; i++) {
			dA_value += W[i * W_x + row] * dZ[i * dZ_x + col];
		}
		dA[row * dA_x + col] = dA_value;
	}
}

__global__ auto linear_update_weights(float* dZ
                                     ,float* A
                                     ,float* W
                                     ,int dZ_x
                                     ,int dZ_y
                                     ,int A_x
                                     ,int A_y
                                     ,float alpha) -> void
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int W_x = A_y;
	int W_y = dZ_y;

	float dW_value = 0.0f;

	if (row < W_y && col < W_x) {
		for (int i = 0; i < dZ_x; i++) {
			dW_value += dZ[row * dZ_x + i] * A[col * A_x + i];
		}
		W[row * W_x + col] = W[row * W_x + col] - alpha * (dW_value / A_x);
	}
}

__global__ auto linear_update_bias(float* dZ
                                  ,float* b
                                  ,int dZ_x
                                  ,int dZ_y
                                  ,int b_x
                                  ,int b_y
                                  ,float alpha) -> void
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x * dZ_y) {
		int dZ_x_new = index % dZ_x;
		int dZ_y_new = index / dZ_x_dim;
		atomicAdd(&b[dZ_y_new], - alpha * (dZ[dZ_y_new * dZ_x + dZ_x_new] / dZ_x));
	}
}

LinearActivationLayer::LinearActivationLayer(std::string title, Dimensions W_dim)
: W(W_dim). b(W_dim.y, 1)
{
    this->name = name;
	b.allocate_memory();
	W.allocate_memory();
	intialize_bias();
	intialize_weights();
}

auto LinearActivationLayer::intialize_bias() -> void
{
    for (int x = 0; x < b.dimensions.x; x++) {
		b[x] = 0;
	}

	b.copy_host_device();
}

auto LinearActivationLayer::intialize_weights() -> void
{
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.dimensions.x; x++) {
		for (int y = 0; y < W.dimensions.y; y++) {
			W[y * W.dimensions.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copy_host_device();
}

auto LinearActivationLayer::backward_propagation_error(Tensor& dZ) -> void
{

}

auto compute_output(Tensor& A) -> void
{

}

auto update_weights(Tensor& dZ, float alpha) -> void
{

}

auto update_bias(Tensor& dZ, float alpha) -> void
{

}

auto LinearActivationLayer::forward_propagation(Tensor& A) -> Tensor&
{

}

auto LinearActivationLayer::backward_propagation(Tensor& dZ, float alpha=0.01) -> Tensor&
{

}

auto LinearActivationLayer::getX() const -> size_t
{

}

auto LinearActivationLayer::getY() const -> size_t
{

}

auto LinearActivationLayer::get_weights() const -> Tensor
{

}

auto LinearActivationLayer::get_bias() const -> Tensor
{

}