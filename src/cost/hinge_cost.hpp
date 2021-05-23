#ifndef HINGE_COST_HPP
#define HINGE_COST_HPP

#include "cost.hpp"
    struct HingeCost : public Cost
    {
        auto cost(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* HINGE_COST_HPP */