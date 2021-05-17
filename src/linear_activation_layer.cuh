#ifndef LINEAR_ACTIVATION_LAYER_HPP
#define LINEAR_ACTIVATION_LAYER_HPP

#include <iostream>
#include "activation_layer.hpp"
#include "Dimensions.hpp"
#include "tensor.hpp"
    struct LinearActivationLayer : ActivationLayer\
    {
        private:    
            const float threshold = 0.01;
            Tensor W;
            Tensor b;
            Tensor Z;
            Tensor A;
            Tensor dA;

            __global__ auto linear_forward_propagation(float* W
                                                      ,float* A
                                                      ,float* Z
                                                      ,float* b
                                                      ,Dimensions W_dim
                                                      ,Dimensions A_dim) -> void;
            
            __global__ auto linear_backward_propagation(float* W
                                                       ,float* dZ
                                                       ,float* dA
                                                       ,Dimensions W_dim
                                                       ,Dimensions dZ_dim) -> void;

            __global__ auto linear_update_weights(float* dZ
                                                 ,float* A
                                                 ,float* W
                                                 ,Dimensions dZ_dim
                                                 ,Dimensions A_dim
                                                 ,float alpha) -> void;
            
            __global__ auto linear_update_bias(float* dZ
                                              ,float* b
                                              ,Dimensions dZ_dim
                                              ,Dimensions b_dim
                                              ,float alpha) -> void;
                                              
            auto intialize_bias() -> void;
            auto intialize_weights() -> void;

            auto backward_propagation_error(Tensor& dZ) -> void;
            auto compute_output(Tensor& A) -> void;
            auto update_weights(Tensor& dZ, float alpha) -> void;
            auto update_bias(Tensor& dZ, float alpha) -> void;
        
        public:
            LinearActivationLayer(std::string title, Dimensions W_dim);

            auto forward_propagation(Tensor& A) -> Tensor&;
            auto backward_propagation(Tensor& dZ, float alpha=0.01) -> Tensor&;

            auto getX() const -> size_t;
            auto getY() const -> size_t;

            auto get_weights() const -> Tensor;
            auto get_bias() const -> Tensor;

    };

#endif /*LINEAR_ACTIVATION_LAYER_HPP*/