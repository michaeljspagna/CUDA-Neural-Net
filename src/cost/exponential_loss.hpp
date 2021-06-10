#ifndef EXPONENTIAL_LOSS_HPP
#define EXPONENTIAL_LOSS_HPP

    #include "loss.hpp"
    #include "../util/tensor.hpp"

    struct ExponentialLoss : public Loss
    {
    public:
        auto loss(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* EXPONENTIAL_LOSS_HPP */