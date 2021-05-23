#ifndef SQUARED_HINGE_COST_HPP
#define SQUARED_HINGE_COST_HPP

#include "cost.hpp"
    struct SquaredHingeCost : public Cost
    {
        auto cost(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* SQUARED_HINGE_COST_HPP */