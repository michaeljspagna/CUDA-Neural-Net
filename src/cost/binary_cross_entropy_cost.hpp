#ifndef BINARY_CROSS_ENTROPY_COST_HPP
#define BINARY_CROSS_ENTROPY_COST_HPP

    #include "cost.hpp"
    #include "../util/tensor.hpp"

    struct BinaryCrossEntropyCost : public Cost
    {
    public:
        auto cost(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* BINARY_CROSS_ENTROPY_COST_HPP */