#ifndef SIGMOID_LAYER_HPP
#define SIGMOID_LAYER_HPP

    #include <string>

    #include "layer.hpp"
    #include "../util/dimension.hpp"
    #include "../util/tensor.hpp"

    struct SigmoidLayer : public Layer
    {
    private:
        Tensor A;
        Tensor Z;
        Tensor Z_error;
    public:
        SigmoidLayer(std::string title);

        auto forward_propagation(Tensor& Z) -> Tensor&;
        auto backward_propagation(Tensor& A_error, float learning_rate=0.01) -> Tensor&;
    };

#endif /* SIGMOID__LAYER_HPP */