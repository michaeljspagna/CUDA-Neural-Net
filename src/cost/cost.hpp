#ifndef COST_HPP
#define COST_HPP

    #include "../util/tensor.hpp"

    struct Cost
    {
        public:
            virtual auto cost(Tensor predictions, Tensor target) -> float = 0;
            virtual auto compute_gradient(Tensor predictions, Tensor target, Tensor gradient) -> Tensor = 0;
    };

#endif /* COST_HPP */