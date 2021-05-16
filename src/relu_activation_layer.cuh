#ifndef RELU_ACTIVATION_LAYER_HPP
#define RELU_ACTIVATION_LAYER_HPP

    #include <string>

    #include "activation_layer.hpp"
    #include "dimensions.hpp"
    #include "tensor.hpp"

    struct ReLUActivationLayer : public ActivationLayer
    {
        private:
            Tensor A;
            Tensor Z;
            Tensor dZ;

            __global__ auto relu_forward_propagation(float* Z
                                                    ,float* A
                                                    ,Dimensions Z_dim) -> void;
            __global__ auto relu_backward_propagation(float* Z
                                                     ,float* dA
                                                     ,float* dZ
                                                     ,Dimensions Z_dim) -> void;
        
        public:
            ReLUActivationLayer(std::string title);

            auto forward_propagation(Tensor& Z) -> Tensor&;
            auto backward_propagation(Tensor& dA, float alpha=0.01) -> Tensor&;
    };
#endif /* RELU_ACTIVATION_LAYER_HPP */