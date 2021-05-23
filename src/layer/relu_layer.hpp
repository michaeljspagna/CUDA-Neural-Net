#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

    #include <string>

    #include "layer.hpp"
    #include "../util/dimension.hpp"
    #include "../util/tensor.hpp"

    struct ReLULayer : public Layer
    {   
        private:
            Tensor A;
            Tensor Z;
            Tensor Z_error;
        public:
            ReLULayer(std::string title);

            auto forward_propagation(Tensor& Z) -> Tensor&;
            auto backward_propagation(Tensor& A_error, float learning_rate=0.01) -> Tensor&;
    };
#endif /* RELU_LAYER_HPP */