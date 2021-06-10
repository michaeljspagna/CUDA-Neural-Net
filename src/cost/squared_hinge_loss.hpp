#ifndef SQUARED_HINGE_LOSS_HPP
#define SQUARED_HINGE_LOSS_HPP

#include "loss.hpp"

    struct SquaredHingeLoss : public Loss
    {
        auto loss(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* SQUARED_HINGE_LOSS_HPP */