#include "tensor.hpp"

Tensor::Tensor(std::size_t x_dimension, std::size_t y_dimension)
    : dimension(x_dimension,y_dimension)
     ,device_data(nullptr), host_data(nullptr)
     ,allocated_on_device(false), allocated_on_host(false){}

Tensor::Tensor(Dimension dimension)
    : Tensor(dimension.x, dimension.y){}

auto Tensor::allocate_memory() -> void
{
    allocate_device_memory();
    allocate_host_memory();
}

auto Tensor::allocate_memory_if_not_allocated(Dimension dimension) -> void
{
    if(!allocated_on_device && !allocated_on_host){
        this->dimension = dimension;
        allocate_memory();
    }
}

auto Tensor::copy_device_to_host() -> void
{
    if(allocated_on_device && allocated_on_host){
        cudaMemcpy(host_data.get(), device_data.get()
                  ,dimension.x * dimension.y * sizeof(float)
                  ,cudaMemcpyDeviceToHost);
    }
}

auto Tensor::copy_host_to_device() -> void
{
    if(allocated_on_device && allocated_on_host){
        cudaMemcpy(device_data.get(), host_data.get()
                  ,dimension.x * dimension.y * sizeof(float)
                  ,cudaMemcpyHostToDevice);
    }
}

auto Tensor::allocate_device_memory() -> void
{
    if(!allocated_on_device){
        float* device_memory = nullptr;
        cudaMalloc(&device_memory
                 ,dimension.x * dimension.y * sizeof(float));
        device_data = std::shared_ptr<float> (d_board
                                             ,[&](float* ptr){ cudaFree(ptr); });
        d_allocated = true;
    }
}

auto Tensor::allocate_host_memory() -> void
{
    if(!allocated_on_host){
        host_data = std::shared_ptr<float>(new float[dimension.x * dimension.y]
                                          ,[&](float* ptr){ delete[] ptr; });
        allocated_on_host = true;
    }
}

auto Tensor::operator[](const int index) -> float& 
{
    return host_data.get()[index];
}

auto Tensor::operator[](const int index) const -> const float& 
{
    return host_data.get()[index];
}
