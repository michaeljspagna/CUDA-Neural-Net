#ifndef SIGMOID_ACTIVATION_LAYER_HPP
#define SIGMOID_ACTIVATION_LAYER_HPP

    #include <string>

    #include "activation_layer.hpp"
    #include "dimensions.hpp"
    #include "tensor.hpp"

    struct SigmoidActivationLayer : public ActivationLayer
    {
        private:
            Tensor A;
            Tensor Z;
            Tensor dZ;

            __device__ auto logistic_sigmoid(float x) -> float;
            __global__ auto sigmoid_forward_propagation(float* Z
                                                       ,float* A
                                                       ,Dimensions Z_dim) -> void;
            __global__ auto sigmoid_backward_propagation(float* Z
                                                        ,float* dA
                                                        ,float* dZ
                                                        ,Dimensions Z_dim) -> void;
        
        public:
            SigmoidActivationLayer(std::string title);

            auto forward_propagation(Tensor& Z) -> Tensor&;
            auto backward_propagation(Tensor& dA, float alpha=0.01) -> Tensor&;
    };
#endif /* SIGMOID_ACTIVATION_LAYER_HPP */