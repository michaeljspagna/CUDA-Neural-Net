#ifndef LOG_LOSS_HPP
#define LOG_LOSS_HPP

    #include "loss.hpp"
    #include "../util/tensor.hpp"

    struct LogLoss : public Loss
    {
    public:
        auto loss(Tensor predictions, Tensor target) -> float;
        auto compute_gradient(Tensor predictions, Tensor target, Tensor Y_change) -> Tensor;
    };

#endif /* LOG_LOSS_HPP */