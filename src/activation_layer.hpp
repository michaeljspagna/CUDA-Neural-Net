#ifndef LAYER_HPP
#define LAYER_HPP

    #include <string>
    #include "tensor.hpp"
    /**
     * Base class for all layer classes
     */
    struct ActivationLayer
    {
        protected:
            std::string title;
        
        public:
            virtual auto forward_propagate(Tensor& A) -> Tensor& = 0;
            virtual auto backward_propagate(Tensor& dZ, float alpha) -> Tensor& = 0;

            auto getTitle() -> std::string 
            {
                return title;
            }
    };
    
#endif /* LAYER_HPP */