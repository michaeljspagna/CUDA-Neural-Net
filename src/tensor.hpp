#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include "dimensions.hpp"

    struct Tensor
    {
        private:
            bool device_allocated;
            bool host_allocated;

            auto allocate_device_memory() -> void;
            auto allocate_host_memory() -> void;

        public:
            /**
             * Holds x, y dimensions
             */
            Dimensions dimensions;
            //count references
            std::shared_ptr<float> device_data;
            std::__shared_ptr<float> host_data;

            Tensor(size_t x=1, size_t y=1)
                :dimensions(Dimensions(x,y))
                ,device_allocated(false)
                ,host_allocated(false){}
            
            Tensor(Dimensions dimensions)
                :Tensor(dimensions.x, dimensions.y){}

            /**
             * Allocates memory on host and device
             */
            auto allocate_memory() -> void;

            /**
             * Checks if memory is already allocated and will allocate if n
             */
            auto allocate_if_not_allocated(Dimensions dimensions) -> void;

            /**
             * Copy data from host to device
             */
            auto copy_host_device() -> void;
            /**
             * Copy data from device to host
             */
            auto copy_device_host() -> void;

            auto operator[](const int index) -> float&;
            auto operator[](const int index) const -> const float&;

    };
#endif /* TENSOR_HPP */