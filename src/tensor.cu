#include "tensor.hpp"

auto Tensor::allocate_device_memory() -> void
{
    if(!device_allocated)
    {
        float* device_memory = nullptr;
        //Allocate device memory
        cudaMalloc(&device_memory, dimensions.x*dimensions.y*sizeof(float));
        //Pass pointer to allocated device memory to shared pointer
        //Pass how to deallocate shared pointer
        device_data = std::shared_ptr<float>(device_memory,
                                             [&](float* ptr){cudaFree(ptr);});
        device_allocated = true;
    }
}

auto Tensor::allocate_host_memory() -> void
{
    if(!host_allocated)
    {
        //Pass ponter to allocate host memory to shared pointer
        //Pass how to delallocate shared pointer
        host_data = std::shared_ptr<float>(new float[dimensions.x*dimensions.y],
                                           [&](float* ptr){delete[] ptr;});
        host_allocated = true;
    }
}

auto Tensor::allocate_memory() -> void
{
    allocate_device_memory();
    allocate_host_memory();
}

auto Tensor::allocate_if_not_allocated(Dimensions dimensions) -> void
{
    if(!device_allocated && !host_allocated)
    {
        this->shape = shape;
        allocate_memory();
    }
}

auto Tensor::copy_host_device() -> void;
{
    cudaMemcpy(device_data.get()
              ,host_data.get()
              ,dimensions.x*dimensions.y*sizeof(float)
              ,cudaMemcpyHostToDevice);
}

auto Tensor::copy_device_host() -> void
{
    cudaMemcpy(host_data.get()
              ,device_data.get()
              ,dimensions.x*dimensions.y*sizeof(float)
              ,cudaMemcpyDeviceToHost);
}

auto Tensor::operator[](const int index) -> float&
{
    return data_host.get()[index];
}

auto Tensor::operator[](const int index) const -> const float&
{
    return host_data.get()[index];
}