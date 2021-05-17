#include "linear_activation_layer.cuh"

LinearActivationLayer::LinearActivationLayer(std::string title, Dimensions W_dim);

__global__ auto linear_forward_propagation(float* W
                                          ,float* A
                                          ,float* Z
                                          ,float* b
                                          ,Dimensions W_dim
                                          ,Dimensions A_dim) -> void
{

}

__global__ auto linear_backward_propagation(float* W
                                           ,float* dZ
                                           ,float* dA
                                           ,Dimensions W_dim
                                           ,Dimensions dZ_dim) -> void
{

}

__global__ auto linear_update_weights(float* dZ
                                     ,float* A
                                     ,float* W
                                     ,Dimensions dZ_dim
                                     ,Dimensions A_dim
                                     ,float alpha) -> void
{

}

__global__ auto linear_update_bias(float* dZ
                                  ,float* b
                                  ,Dimensions dZ_dim
                                  ,Dimensions b_dim
                                  ,float alpha) -> void
{

}

auto LinearActivationLayer::intialize_bias() -> void
{

}

auto intialize_weights() -> void
{

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