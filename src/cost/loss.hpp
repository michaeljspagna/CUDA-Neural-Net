#ifndef LOSS_HPP
#define LOSS_HPP

    #include "../util/tensor.hpp"

    struct Loss
    {
        public:
            virtual auto loss(Tensor predictions, Tensor target) -> float = 0;
            virtual auto compute_gradient(Tensor predictions, Tensor target, Tensor gradient) -> Tensor = 0;
    };

#endif /* LOSS_HPP */