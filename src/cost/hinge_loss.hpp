#ifndef HINGE_LOSS_HPP
#define HINGE_LOSS_HPP

#include "loss.hpp"
    struct HingeLoss : public Loss
    {
        auto loss(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* HINGE_LOSS_HPP */