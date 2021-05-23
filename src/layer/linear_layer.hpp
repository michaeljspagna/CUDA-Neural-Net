#ifndef LINEAR_ACTIVATION_LAYER_HPP
#define LINEAR_ACTIVATION_LAYER_HPP

#include <iostream>
#include "layer.hpp"
#include "../util/dimension.hpp"
#include "../util/tensor.hpp"

    struct LinearLayer : public Layer
    {   
        private:
            const float weights_init_threshold = 0.01;

            Tensor weights;
            Tensor bias;
            Tensor A;
            Tensor A_error;
            Tensor Z;

            auto initialize_bias() -> void;
            auto initalize_weights() -> void;
            auto compute_error(Tensor& Z_error) -> void;
            auto compute_output(Tensor& A) -> void;
            auto update_weights(Tensor& Z_error, float learning_rate) -> void;
            auto update_bias(Tensor& Z_error, float learning_rate) -> void;

        public:
            LinearLayer(std::string title, Dimension weights_dimension);

            auto forward(Tensor& A) -> Tensor&;
            auto backprop(Tensor& Z_error, float learning_rate = 0.01) -> Tensor&;
            auto get_weights_x() const -> int;
            auto get_weights_y() const -> int;
            auto get_weights() const -> Tensor;
            auto get_bias() const -> Tensor;
        };

#endif /*LINEAR_ACTIVATION_LAYER_HPP*/