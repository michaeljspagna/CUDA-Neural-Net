#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <memory>
#include "dimension.hpp"

    struct Tensor
    {
        public:
            Dimension dimension;

            std::shared_ptr<float> device_data;
            std::shared_ptr<float> host_data;

            Tensor(std::size_t x_dimension = 1, std::size_t y_dimension = 1);
            Tensor(Dimension dimension);

            auto allocate_memory() -> void;
            auto allocate_memory_if_not_allocated(Dimension dimension) -> void;

            auto copy_device_to_host() -> void;
            auto copy_host_to_device() -> void;

            auto operator[](const int index) -> float&;
            auto operator[](const int index) const -> const float&;

        private:
            bool allocated_on_device;
            bool allocated_on_host;

            auto allocate_device_memory() -> void;
            auto allocate_host_memory() -> void;

    };
#endif /* TENSOR_HPP */